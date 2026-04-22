"""
Full test suite — run with: pytest tests/ -v
"""
import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oceanus.physics import GameState, GhostNet, ASVState, PolicyInbox, GRID_SIZE
from oceanus.models import (
    OceanusEnv, parse_asv_action, parse_policy_action,
    build_asv_observation, build_policy_observation,
    build_asv_prompt, build_policy_prompt, RewardEngine,
)
from oceanus.adversary import AdversaryAgent
from oceanus.runner import OceanusRunner, MockASVAgent, MockPolicyAgent


# ─────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────

class TestPhysics:
    def test_grid_init(self):
        state = GameState(seed=0)
        assert state.grid.shape == (GRID_SIZE, GRID_SIZE)
        assert len(state.ghost_nets) == 15

    def test_asv_movement(self):
        asv = ASVState("ASV-1", row=5, col=5)
        asv.move("north"); assert asv.row == 4 and asv.col == 5
        asv.move("east");  assert asv.row == 4 and asv.col == 6
        asv.move("south"); assert asv.row == 5 and asv.col == 6
        asv.move("west");  assert asv.row == 5 and asv.col == 5

    def test_asv_boundary_clamping(self):
        asv = ASVState("ASV-1", row=0, col=0)
        asv.move("north"); assert asv.row == 0
        asv.move("west");  assert asv.col == 0
        asv2 = ASVState("ASV-2", row=GRID_SIZE-1, col=GRID_SIZE-1)
        asv2.move("south"); assert asv2.row == GRID_SIZE - 1
        asv2.move("east");  assert asv2.col == GRID_SIZE - 1

    def test_net_drift(self):
        net = GhostNet(row=10, col=10)
        net.drift((1, 0))
        assert 0 <= net.row < GRID_SIZE
        assert 0 <= net.col < GRID_SIZE

    def test_clean_net(self):
        state = GameState(seed=0)
        net = GhostNet(row=5, col=5, density=1.0)
        state.ghost_nets.append(net)
        state.grid[5, 5] = 1.0
        initial_count = len(state.ghost_nets)
        cleaned = state.clean_net_at(5, 5)
        assert cleaned is True
        assert len(state.ghost_nets) == initial_count - 1
        assert state.grid[5, 5] == 0.0
        assert state.total_cleaned >= 1

    def test_clean_empty_cell(self):
        state = GameState(seed=0)
        state.ghost_nets = []
        state.grid[:] = 0
        cleaned = state.clean_net_at(10, 10)
        assert cleaned is False

    def test_sonar_reading(self):
        state = GameState(seed=0)
        asv = state.asvs["ASV-1"]
        sonar = state.get_sonar_reading(asv)
        assert len(sonar) == 3
        assert len(sonar[0]) == 3
        assert sonar[1][1] == "self"

    def test_biodiversity_update(self):
        state = GameState(seed=0)
        state.ghost_nets = []
        state.total_cleaned = 50
        state.update_biodiversity()
        assert state.biodiversity_index > 50

    def test_spawn_rate_update(self):
        state = GameState(seed=0)
        state.treaty_status = "Tagging Mandate Active"
        state.update_spawn_rate()
        assert state.spawn_interval == 15
        assert state.net_spawn_rate == "Low"

    def test_spawn_rate_medium(self):
        state = GameState(seed=0)
        state.treaty_status = "Treaty Under Review"
        state.update_spawn_rate()
        assert state.spawn_interval == 10

    def test_tick_advances_step(self):
        state = GameState(seed=0)
        assert state.step_count == 0
        state.tick()
        assert state.step_count == 1

    def test_policy_inbox_pop(self):
        inbox = PolicyInbox()
        inbox.add_message("Alice", "Angry", "Hello")
        inbox.add_message("Bob", "Neutral", "Hi")
        msg = inbox.pop_message("Alice")
        assert msg is not None
        assert msg["from"] == "Alice"
        assert inbox.unanswered_count == 1
        # Bob still in inbox
        assert len(inbox.messages) == 1
        assert list(inbox.messages)[0]["from"] == "Bob"

    def test_policy_inbox_pop_missing(self):
        inbox = PolicyInbox()
        inbox.add_message("Alice", "Angry", "Hello")
        msg = inbox.pop_message("NonExistent")
        assert msg is None
        assert inbox.unanswered_count == 1


# ─────────────────────────────────────────────
# Action Parsers
# ─────────────────────────────────────────────

class TestParsers:
    def test_valid_asv_move(self):
        a, v = parse_asv_action('{"intent": "move", "direction": "north"}')
        assert v and a["intent"] == "move" and a["direction"] == "north"

    def test_valid_asv_clean(self):
        a, v = parse_asv_action('{"intent": "clean"}')
        assert v and a["intent"] == "clean"

    def test_valid_asv_scan(self):
        a, v = parse_asv_action('{"intent": "scan"}')
        assert v and a["intent"] == "scan"

    def test_valid_asv_broadcast(self):
        a, v = parse_asv_action('{"intent": "broadcast", "message": "Net at row 5"}')
        assert v and a["message"] == "Net at row 5"

    def test_invalid_json(self):
        a, v = parse_asv_action("I think I should go north")
        assert not v and a is None

    def test_empty_string(self):
        a, v = parse_asv_action("")
        assert not v

    def test_none_input(self):
        a, v = parse_asv_action(None)
        assert not v

    def test_json_embedded_in_text(self):
        a, v = parse_asv_action('Thinking... {"intent": "move", "direction": "east"} done.')
        assert v and a["direction"] == "east"

    def test_invalid_direction(self):
        a, v = parse_asv_action('{"intent": "move", "direction": "diagonal"}')
        assert not v

    def test_broadcast_empty_message(self):
        a, v = parse_asv_action('{"intent": "broadcast", "message": ""}')
        assert not v

    def test_valid_policy_propose(self):
        a, v = parse_policy_action(
            '{"intent": "propose_treaty", "target": "Fleet_Manager", "content": "50% subsidy"}'
        )
        assert v and a["intent"] == "propose_treaty"

    def test_valid_policy_accept(self):
        a, v = parse_policy_action(
            '{"intent": "accept_treaty", "target": "Port_Authority", "content": "Agreed"}'
        )
        assert v

    def test_valid_policy_reply(self):
        a, v = parse_policy_action(
            '{"intent": "reply_email", "target": "Fisher_Bob", "content": "We hear you."}'
        )
        assert v and a["intent"] == "reply_email"

    def test_policy_missing_target(self):
        a, v = parse_policy_action('{"intent": "reply_email", "content": "Hello"}')
        assert not v

    def test_missing_intent(self):
        a, v = parse_asv_action('{"direction": "north"}')
        assert not v


# ─────────────────────────────────────────────
# Observations
# ─────────────────────────────────────────────

class TestObservations:
    def setup_method(self):
        self.env = OceanusEnv(seed=42)
        self.env.reset()

    def test_asv_observation_keys(self):
        obs = build_asv_observation(self.env.state, "ASV-1")
        for key in ("agent_id", "sonar_3x3", "battery", "position", "step", "biodiversity_index"):
            assert key in obs
        assert obs["agent_id"] == "ASV-1"

    def test_policy_observation_keys(self):
        obs = build_policy_observation(self.env.state, "Port_Authority")
        for key in ("agent_id", "current_policy_status", "inbox", "net_spawn_rate"):
            assert key in obs

    def test_asv_prompt_has_json_instruction(self):
        obs = build_asv_observation(self.env.state, "ASV-1")
        prompt = build_asv_prompt(obs)
        assert "intent" in prompt and "JSON" in prompt

    def test_policy_prompt_has_treaty_info(self):
        obs = build_policy_observation(self.env.state, "Port_Authority")
        prompt = build_policy_prompt(obs)
        assert "treaty" in prompt.lower()


# ─────────────────────────────────────────────
# Reward Engine
# ─────────────────────────────────────────────

class TestRewardEngine:
    def setup_method(self):
        self.engine = RewardEngine()

    def test_clean_reward(self):
        r = self.engine.compute_asv_reward({"intent": "clean"}, True, True, False)
        assert r == 5.0

    def test_invalid_json_penalty(self):
        r = self.engine.compute_asv_reward(None, False, False, False)
        assert r == -1.0

    def test_move_energy_cost(self):
        r = self.engine.compute_asv_reward({"intent": "move"}, True, False, False)
        assert r == -0.1

    def test_broadcast_reward(self):
        r = self.engine.compute_asv_reward({"intent": "broadcast"}, True, False, True)
        assert r == 0.5

    def test_treaty_reward(self):
        r = self.engine.compute_policy_reward(
            {"intent": "accept_treaty"}, True, True, False, False
        )
        assert r == 10.0

    def test_email_ignored_penalty(self):
        r = self.engine.compute_policy_reward(None, False, False, False, True)
        assert r <= -5.0

    def test_global_reward_recovery(self):
        env = OceanusEnv(seed=42)
        env.reset()
        env.state.done = True
        env.state.biodiversity_index = 80.0
        assert self.engine.compute_global_reward(env.state) == 100.0

    def test_global_reward_partial(self):
        env = OceanusEnv(seed=42)
        env.reset()
        env.state.done = True
        env.state.biodiversity_index = 60.0
        assert self.engine.compute_global_reward(env.state) == 40.0

    def test_global_reward_collapse(self):
        env = OceanusEnv(seed=42)
        env.reset()
        env.state.done = True
        env.state.biodiversity_index = 10.0
        assert self.engine.compute_global_reward(env.state) == -20.0

    def test_no_global_reward_mid_episode(self):
        env = OceanusEnv(seed=42)
        env.reset()
        env.state.done = False
        env.state.biodiversity_index = 90.0
        assert self.engine.compute_global_reward(env.state) == 0.0


# ─────────────────────────────────────────────
# Full Environment
# ─────────────────────────────────────────────

class TestOceanusEnv:
    def test_reset_returns_all_agents(self):
        env = OceanusEnv(seed=42)
        obs = env.reset()
        expected = {"ASV-1", "ASV-2", "ASV-3", "ASV-4", "Port_Authority", "Fleet_Manager"}
        assert set(obs.keys()) == expected

    def test_step_returns_correct_structure(self):
        env = OceanusEnv(seed=42)
        env.reset()
        actions = {
            "ASV-1": '{"intent": "scan"}',
            "ASV-2": '{"intent": "move", "direction": "north"}',
            "ASV-3": '{"intent": "scan"}',
            "ASV-4": '{"intent": "scan"}',
            "Port_Authority": '{"intent": "reply_email", "target": "Fisher_Bob", "content": "We hear you."}',
            "Fleet_Manager": '{"intent": "propose_treaty", "target": "Port_Authority", "content": "50pct subsidy"}',
        }
        obs, rewards, done, info = env.step(actions)
        assert "__total__" in rewards
        assert "step" in info and "biodiversity" in info and "active_nets" in info
        assert isinstance(done, bool)

    def test_episode_runs_to_completion(self):
        env = OceanusEnv(seed=42, max_steps=10)
        adversary = AdversaryAgent(inject_interval=5)
        runner = OceanusRunner(env, adversary, use_mock=True, verbose=False)
        summary = runner.run_episode(episode_id=0)
        assert summary["total_steps"] == 10
        assert "total_reward" in summary
        assert "biodiversity_final" in summary

    def test_treaty_reduces_spawn_rate(self):
        env = OceanusEnv(seed=42)
        env.reset()
        assert env.state.spawn_interval == 5
        env.state.treaty_status = "Tagging Mandate Active"
        env.state.update_spawn_rate()
        assert env.state.spawn_interval == 15

    def test_invalid_actions_dont_crash(self):
        env = OceanusEnv(seed=42)
        env.reset()
        actions = {
            "ASV-1": "not json at all",
            "ASV-2": "",
            "ASV-3": '{"intent": "scan"}',
            "ASV-4": '{"intent": "scan"}',
            "Port_Authority": '{"intent": "reply_email", "target": "Fisher_Bob", "content": "ok"}',
            "Fleet_Manager": '{"intent": "scan"}',
        }
        obs, rewards, done, info = env.step(actions)
        assert "__total__" in rewards
        # Invalid actions get penalty but don't crash
        assert rewards["ASV-1"] == -1.0
        assert rewards["ASV-2"] == -1.0

    def test_asv_nets_cleaned_counter(self):
        env = OceanusEnv(seed=42)
        env.reset()
        # Place a net at ASV-1's position
        from oceanus.physics import GhostNet
        asv = env.state.asvs["ASV-1"]
        net = GhostNet(row=asv.row, col=asv.col, density=1.0)
        env.state.ghost_nets.append(net)
        env.state.grid[asv.row, asv.col] = 1.0

        actions = {aid: '{"intent": "scan"}' for aid in env.state.asvs}
        actions["ASV-1"] = '{"intent": "clean"}'
        actions["Port_Authority"] = '{"intent": "reply_email", "target": "Fisher_Bob", "content": "ok"}'
        actions["Fleet_Manager"] = '{"intent": "reply_email", "target": "Port_Authority_Agent", "content": "ok"}'

        obs, rewards, done, info = env.step(actions)
        assert rewards["ASV-1"] == 5.0  # clean reward
        assert env.state.asvs["ASV-1"].nets_cleaned == 1


# ─────────────────────────────────────────────
# Adversary
# ─────────────────────────────────────────────

class TestAdversary:
    def test_inject_at_interval(self):
        adv = AdversaryAgent(inject_interval=20, seed=0)
        assert not adv.should_inject(0)
        assert adv.should_inject(20)
        assert adv.should_inject(40)
        assert not adv.should_inject(21)

    def test_difficulty_scales(self):
        adv = AdversaryAgent(inject_interval=20, seed=0)
        adv.scale_difficulty(0, 200)
        assert adv.difficulty == 1.0
        adv.scale_difficulty(100, 200)
        assert adv.difficulty > 1.0
        adv.scale_difficulty(200, 200)
        assert adv.difficulty == 3.0

    def test_inject_returns_list(self):
        adv = AdversaryAgent(inject_interval=20, seed=42)
        state = GameState(seed=42)
        state.step_count = 20
        events = adv.inject(state)
        assert isinstance(events, list)

    def test_storm_spawns_nets(self):
        adv = AdversaryAgent(inject_interval=20, seed=0)
        state = GameState(seed=0)
        initial = len(state.ghost_nets)
        adv._apply_storm(state)
        assert len(state.ghost_nets) > initial

    def test_rogue_trawler_in_bounds(self):
        adv = AdversaryAgent(inject_interval=20, seed=0)
        state = GameState(seed=0)
        adv._apply_rogue_trawler(state)
        for net in state.ghost_nets:
            assert 0 <= net.row < GRID_SIZE
            assert 0 <= net.col < GRID_SIZE

    def test_zero_max_steps_no_crash(self):
        adv = AdversaryAgent(inject_interval=20, seed=0)
        adv.scale_difficulty(0, 0)  # should not divide by zero
        assert adv.difficulty == 1.0


# ─────────────────────────────────────────────
# Mock Agents
# ─────────────────────────────────────────────

class TestMockAgents:
    def test_mock_asv_returns_valid_json(self):
        agent = MockASVAgent("ASV-1")
        obs = {
            "sonar_3x3": [
                ["empty", "empty", "empty"],
                ["empty", "self", "empty"],
                ["empty", "empty", "empty"],
            ],
            "position": {"row": 5, "col": 5, "sector": "B2"},
        }
        action_str = agent.act("prompt", obs)
        a, v = parse_asv_action(action_str)
        assert v

    def test_mock_asv_cleans_when_net_at_position(self):
        agent = MockASVAgent("ASV-1")
        obs = {
            "sonar_3x3": [
                ["empty", "empty", "empty"],
                ["empty", "net", "empty"],  # net at center = our position
                ["empty", "empty", "empty"],
            ],
            "position": {"row": 5, "col": 5, "sector": "B2"},
        }
        action_str = agent.act("prompt", obs)
        a, v = parse_asv_action(action_str)
        assert v and a["intent"] == "clean"

    def test_mock_asv_moves_toward_net(self):
        agent = MockASVAgent("ASV-1")
        obs = {
            "sonar_3x3": [
                ["net", "empty", "empty"],  # net to north-west
                ["empty", "self", "empty"],
                ["empty", "empty", "empty"],
            ],
            "position": {"row": 5, "col": 5, "sector": "B2"},
        }
        action_str = agent.act("prompt", obs)
        a, v = parse_asv_action(action_str)
        assert v and a["intent"] == "move"

    def test_mock_policy_proposes_treaty(self):
        agent = MockPolicyAgent("Port_Authority")
        obs = {
            "inbox": [],
            "current_policy_status": "No Tagging Mandate",
            "net_spawn_rate": "High",
            "biodiversity_index": 60.0,
            "unanswered_emails": 0,
            "step": 5,
        }
        action_str = agent.act("prompt", obs)
        a, v = parse_policy_action(action_str)
        assert v and a["intent"] == "propose_treaty"

    def test_mock_policy_accepts_treaty(self):
        agent = MockPolicyAgent("Fleet_Manager")
        obs = {
            "inbox": [],
            "current_policy_status": "Treaty Proposed",
            "net_spawn_rate": "High",
            "biodiversity_index": 60.0,
            "unanswered_emails": 0,
            "step": 10,
        }
        action_str = agent.act("prompt", obs)
        a, v = parse_policy_action(action_str)
        assert v and a["intent"] == "accept_treaty"
