import unittest

import numpy as np

from rcm.representation import IdentityAdapter, RepresentationController, ProposalType


class RepresentationControllerTests(unittest.TestCase):
    def test_proposal_can_expand_and_accept(self):
        controller = RepresentationController(adapter=IdentityAdapter(), current_dim=4, min_dim=2, max_dim=8)
        state = np.array([0.2, -0.1, 0.4, 0.1], dtype=float)
        proposal = controller.propose(state, objective=0.7, step=15)
        self.assertEqual(proposal.proposal_type, ProposalType.EXPAND)
        accepted, delta, new_state = controller.evaluate(proposal, objective=0.7, current_state=state, step=15)
        self.assertTrue(accepted)
        self.assertGreaterEqual(delta, 0.0)
        self.assertEqual(len(new_state), proposal.target_dim)

    def test_cooldown_and_collapse_block_changes(self):
        controller = RepresentationController(adapter=IdentityAdapter(), current_dim=4, min_dim=2, max_dim=8, cooldown_steps=10)
        state = np.array([0.2, -0.1, 0.4, 0.1], dtype=float)
        controller.history.append(type("Entry", (), {"accepted": True})())
        controller.last_change_step = 0
        proposal = controller.propose(state, objective=0.7, step=5)
        self.assertEqual(proposal.proposal_type, ProposalType.NO_CHANGE)

    def test_contract_and_rollback_are_supported(self):
        controller = RepresentationController(adapter=IdentityAdapter(), current_dim=4, min_dim=2, max_dim=8)
        state = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        proposal = controller.propose(state, objective=0.5, step=15)
        self.assertEqual(proposal.proposal_type, ProposalType.CONTRACT)
        self.assertLess(proposal.target_dim, 4)


if __name__ == "__main__":
    unittest.main()
