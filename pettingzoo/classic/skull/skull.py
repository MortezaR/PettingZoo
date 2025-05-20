# noqa: D212, D415
"""
Skull (simplified)

This is a simplified implementation of the board game Skull for two players.
The goal is to win a single bidding round by successfully revealing a number
of discs without revealing a skull.

| Import             | `from pettingzoo.classic import skull_v0` |
|--------------------|--------------------------------------------|
| Actions            | Discrete                                   |
| Parallel API       | Yes                                        |
| Manual Control     | Yes                                        |
| Agents             | `agents= ['player_0', 'player_1']`          |
| Agents             | 2                                          |
| Action Shape       | (1,)                                       |
| Action Values      | Discrete(5)                                |
| Observation Shape  | (8,)                                       |
| Observation Values | Mixed                                      |

This environment is intended only as a minimal example and does not implement
all of the rules of Skull. Each player has three roses and one skull disc. At
the start of the round players alternate placing discs face down. After at least
one disc from each player is placed, a player may start the bidding. During the
bidding phase players may raise the bid or pass. Once all but the highest bidder
have passed, the highest bidder attempts to reveal the declared number of discs.
They must reveal their own stack first and then the opponent's. Revealing a skull
ends the game with a loss for the bidder; otherwise they win.

### Actions
0. Place a rose
1. Place a skull
2. Bid/Raise
3. Pass
4. Reveal opponent disc (only used in the reveal phase when bidder has revealed
   all personal discs)

### Observation Space
The observation is a dictionary with two elements:
`observation` - a vector of integers describing the state
`action_mask` - a binary vector of length 5 indicating legal actions

The observation vector contains:
[own_roses, own_skull, own_stack_size, opp_stack_size,
 phase, highest_bid, highest_bidder, discs_left_to_reveal]

### Rewards
The winning player receives +1 and the loser -1. Illegal moves
terminate the game with -1 reward for the offending agent.

### Version History
* v0: Initial version
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


ROSE = 0
SKULL = 1


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human"],
        "name": "skull_v0",
        "is_parallelizable": True,
        "render_fps": 1,
    }

    def __init__(self, render_mode: str | None = None):
        EzPickle.__init__(self, render_mode)
        super().__init__()
        self.render_mode = render_mode

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self._agent_selector = AgentSelector(self.agents)

        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=8, shape=(8,), dtype=np.int8),
                    "action_mask": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                }
            )
            for agent in self.agents
        }

        self._screen = None
        self.clock = None
        self._font = None
        self.last_revealed: Optional[int] = None

        self.reset()

    # game state helpers -------------------------------------------------
    def _current_index(self) -> int:
        return self.agents.index(self.agent_selection)

    def _opponent_index(self, idx: int) -> int:
        return 1 - idx

    # mask ---------------------------------------------------------------
    def _get_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(5, dtype=np.int8)
        idx = self.agents.index(agent)
        opp = self._opponent_index(idx)
        if self.phase == 0:  # placing
            if self.roses[idx] > 0:
                mask[0] = 1
            if self.skulls[idx] > 0:
                mask[1] = 1
            if len(self.stacks[idx]) > 0 and len(self.stacks[opp]) > 0:
                mask[2] = 1
        elif self.phase == 1:  # bidding
            if not self.passed[agent]:
                mask[2] = 1
            mask[3] = 1
        elif self.phase == 2:  # reveal
            if agent == self.reveal_player and self.reveal_count >= len(self.stacks[idx]):
                if len(self.stacks[opp]) > 0:
                    mask[4] = 1
        return mask

    # observation --------------------------------------------------------
    def observe(self, agent: str):
        idx = self.agents.index(agent)
        opp = self._opponent_index(idx)
        obs = np.array(
            [
                self.roses[idx],
                self.skulls[idx],
                len(self.stacks[idx]),
                len(self.stacks[opp]),
                self.phase,
                self.highest_bid,
                self.highest_bidder if self.highest_bidder is not None else 2,
                max(self.highest_bid - self.reveal_count, 0) if self.phase == 2 else 0,
            ],
            dtype=np.int8,
        )
        return {"observation": obs, "action_mask": self._get_mask(agent)}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # core game logic ----------------------------------------------------
    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        idx = self._current_index()
        opp = self._opponent_index(idx)
        agent = self.agent_selection
        self.last_revealed = None

        if self.phase == 0:  # placing
            if action == 0:
                assert self.roses[idx] > 0
                self.roses[idx] -= 1
                self.stacks[idx].append(ROSE)
            elif action == 1:
                assert self.skulls[idx] > 0
                self.skulls[idx] -= 1
                self.stacks[idx].append(SKULL)
            elif action == 2 and len(self.stacks[idx]) > 0 and len(self.stacks[opp]) > 0:
                self.phase = 1
                self.highest_bid = len(self.stacks[idx]) + len(self.stacks[opp])
                self.highest_bidder = idx
                self.passed = {a: False for a in self.agents}
            else:
                assert False, "Illegal action"
        elif self.phase == 1:  # bidding
            if action == 2 and not self.passed[agent]:
                self.highest_bid += 1
                self.highest_bidder = idx
            elif action == 3:
                self.passed[agent] = True
                if all(self.passed[a] or self.agents.index(a) == self.highest_bidder for a in self.agents):
                    self.phase = 2
                    self.reveal_player = self.agents[self.highest_bidder]
                    self.reveal_count = 0
            else:
                assert False, "Illegal action"
        elif self.phase == 2:  # reveal
            assert agent == self.reveal_player
            if self.reveal_count < len(self.stacks[idx]):
                disc = self.stacks[idx].pop()
            else:
                assert action == 4 and len(self.stacks[opp]) > 0
                disc = self.stacks[opp].pop()
            self.last_revealed = disc
            self.reveal_count += 1
            if disc == SKULL:
                self.rewards[agent] -= 1
                self.rewards[self.agents[opp]] += 1
                self.terminations = {a: True for a in self.agents}
            elif self.reveal_count >= self.highest_bid:
                self.rewards[agent] += 1
                self.rewards[self.agents[opp]] -= 1
                self.terminations = {a: True for a in self.agents}
        else:
            assert False, "Invalid phase"

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.roses = [3, 3]
        self.skulls = [1, 1]
        self.stacks: List[List[int]] = [[], []]
        self.phase = 0  # 0 placing, 1 bidding, 2 reveal
        self.highest_bid = 0
        self.highest_bidder: Optional[int] = None
        self.passed = {a: False for a in self.agents}
        self.reveal_player: Optional[str] = None
        self.reveal_count = 0
        self.last_revealed = None
        self.last_revealed = None

    def render(self):
        if self.render_mode != "human":
            print(
                f"Stacks: {self.stacks}, Roses: {self.roses}, Skulls: {self.skulls}"
            )
            print(
                f"Phase: {self.phase}, Bid: {self.highest_bid}, Bidder: {self.highest_bidder}"
            )
            return

        import pygame

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((500, 300))
            pygame.display.set_caption("Skull")
            self._font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()

        self._screen.fill((255, 255, 255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None
                return

        def draw_stack(stack, x, y):
            for i, _ in enumerate(stack):
                pygame.draw.rect(
                    self._screen,
                    (200, 200, 200),
                    (x + i * 45, y, 40, 40),
                )

        draw_stack(self.stacks[0], 40, 120)
        draw_stack(self.stacks[1], 40, 200)

        lines = [
            f"P0 roses:{self.roses[0]} skulls:{self.skulls[0]}",
            f"P1 roses:{self.roses[1]} skulls:{self.skulls[1]}",
            f"Phase:{self.phase} Bid:{self.highest_bid} Bidder:{self.highest_bidder}",
            f"Current:{self.agent_selection}",
        ]
        if self.last_revealed is not None:
            lines.append(
                "Last reveal: " + ("ROSE" if self.last_revealed == ROSE else "SKULL")
            )

        for i, line in enumerate(lines):
            img = self._font.render(line, True, (0, 0, 0))
            self._screen.blit(img, (40, 20 + i * 20))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pass


def manual_control():
    """Launch a simple window for playing Skull via the keyboard."""
    import pygame

    env_inst = env(render_mode="human")
    env_inst.reset()

    mapping_p0 = {
        pygame.K_w: 0,
        pygame.K_a: 1,
        pygame.K_s: 2,
        pygame.K_d: 3,
        pygame.K_j: 4,
    }

    mapping_p1 = {
        pygame.K_UP: 0,
        pygame.K_LEFT: 1,
        pygame.K_DOWN: 2,
        pygame.K_RIGHT: 3,
        pygame.K_k: 4,
    }

    running = True
    while running:
        for agent in env_inst.agent_iter():
            obs, _, term, trunc, _ = env_inst.last()
            action = None
            while action is None and running:
                env_inst.render()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            action = 0
                            break
                        mapping = mapping_p0 if agent == "player_0" else mapping_p1
                        if event.key in mapping and obs["action_mask"][mapping[event.key]]:
                            action = mapping[event.key]
            if not running:
                break
            env_inst.step(action)
            if term or trunc:
                env_inst.render()
                env_inst.reset()
                break

    env_inst.close()
    pygame.quit()


