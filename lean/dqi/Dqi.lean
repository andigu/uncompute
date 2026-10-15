-- Formal development supporting the DQI decoder error bounds.
--
-- `Dqi.Basic`        patterns, syndromes, the dual code, and injectivity of the
--                    syndrome map on the decodable set
-- `Dqi.Failure`      the failure model: registers, labels, branch-diagonality,
--                    the failure set, and the DQI probability mass
-- `Dqi.Characters`   additive-character orthogonality over a finite field, the
--                    delta relation behind the syndrome-pullback step
-- `Dqi.LayerState`  normalized fixed-Hamming-weight layer states and their
--                    orthonormality, plus the binomial coefficients of the shell formula
-- `Dqi.Objective`   the objective as an operator: the character shift structure
--                    behind the syndrome pullback, and bounded self-adjoint
--                    multiplication operators
-- `Dqi.Reachable`   coordinate restriction: the projector onto the outputs a
--                    perturbation can populate, and the reachable-residual
--                    Cauchy-Schwarz step
-- `Dqi.Fourier`     the bridge from the position basis to the pattern basis, and
--                    the two amplitude facts LayerState takes as hypotheses
-- `Dqi.Amplitudes`  joins the two: the real DQI amplitude family, with the layer
--                    states' normalization hypotheses discharged rather than assumed
-- `Dqi.SiteOp`      the two rows of the single-site constraint operator in the
--                    pattern basis: the entries of the 2x2 shell matrix
-- `Dqi.Shell`       THE SHELL IDENTITY: the objective connects only neighbouring
--                    Hamming-weight layers, with the tridiagonal coefficients.
--                    This is what makes the ideal profile an eigenvector within
--                    the decodable subspace, hence eigenstate protection.
-- `Dqi.Pullback`    the syndrome-pullback support conditions: at strict distance
--                    no dual codeword is short enough to create an alias, so the
--                    objective's only matrix elements are the obvious ones
-- `Dqi.Residual`    the residual decomposition: the objective acts tridiagonally
--                    inside the decodable subspace, and EXACTLY ONE direction
--                    escapes it (the next weight layer). Defines sigma_edge and
--                    the exact finite-size objective value lamF.
-- `Dqi.Channel`     the blindness bound: a perturbation confined to one component's
--                    coordinates cannot see the other, plus mass-as-restricted-norm
--                    and the per-channel splitting
-- `Dqi.Embed`       the pattern-to-syndrome translation: decodable syndromes and
--                    edge syndromes are DISJOINT at strict distance. This is what
--                    lets Channel's blindness bound apply to sign/leakage errors.
-- `Dqi.Flat`        the flat amplitude constant is FORCED by normalization, not
--                    assumed: given only that the nonzero magnitudes are equal,
--                    each must be (q-1)^(-1/2)
-- `Dqi.Counting`    the relocation-bound arithmetic: Hamming shell sizes, flat
--                    amplitudes spreading a layer state uniformly, and the
--                    labels-times-amplitude norm bound
-- `Dqi.Degradation` THE ASSEMBLED BOUND: master perturbation bound + reachable
--                    residual, in the two cases of the appendix's proposition --
--                    blind (sign/leakage) and reaching (relocation)
-- `Dqi.Bridge`      branch indexing to physical-register indexing: the mass on the
--                    registers a set of branches reaches equals the mass those
--                    branches carried. Identifies the mass the bound consumes with
--                    the mass the validation protocol estimates.
-- `Dqi.Boundary`    counting for the boundary regime (decoding radius exactly meets
--                    half the dual distance): how many minimum-weight codewords are
--                    compatible with an output, and the shape of the R_C bound
-- `Dqi.ShellOp`     the objective packaged as a bounded self-adjoint operator:
--                    each site is a reflection about a projection, so the norm is
--                    at most m. This is what the abstract bound instantiates at.
-- `Dqi.Main`        THE MASTER THEOREM (strict distance): the abstract bound
--                    instantiated at the real objective and the real DQI state.
-- `Dqi.MainBoundary` the boundary variant: the residual split and the widened
--                    decodable bound (sigmaDec + R_C) enter as cited hypotheses.
-- `Dqi.FourierN`    the n-variable character transform with inversion and
--                    unitarity: the change of basis between the measurement
--                    basis and the syndrome basis
-- `Dqi.Phys`        the physical objective (diagonal in the measurement basis,
--                    conjugated to the syndrome basis) and the pattern-to-syndrome
--                    pushforward with its metric transport
-- `Dqi.Intertwine`  THE PULLBACK INTERTWINING: the measured objective, conjugated
--                    to the syndrome basis, equals the pattern-space shell operator
--                    transported along the pushforward -- unconditionally. Distance
--                    enters only through toSyn's metric transport, never here.
-- `Dqi.Taxonomy`    the exclusive failure-channel partition (relocation, leakage,
--                    sign) defined from branch-diagonal data, with the support fact
--                    behind blindness and the mass partition mu = mu_r + mu_L + mu_s
-- `Dqi.PhysFull`    the objective lifted to the full register space (trivial on
--                    ancilla and record), the clean-register embedding, and the
--                    lifted intertwining
-- `Dqi.Ewire`       the relocation reach computed: labels x flat amplitude x
--                    collisions, in the physical register space
-- `Dqi.CrossPhys`   the sharp relocation cross-term: ideal images and non-clean
--                    channels are exactly orthogonal to the escaping component, so
--                    only the clean-relocation mass multiplies the reach
-- `Dqi.Packing`      Fisher-type packing bound (how many patterns can share a syndrome)
-- `Dqi.Perturbation` the master degradation bound: a norm-preserving perturbation
--                    moves a self-adjoint expectation only at second order, apart
--                    from a cross term controlled by the reachable residual
-- `Dqi.Complete`    THE END-TO-END THEOREM: dqi_end_to_end, the degradation bound
--                    with every internal hypothesis discharged -- hypotheses are only the
--                    instance data, the cited code facts, bent constancy, the
--                    profile, the two branch-diagonal oracles, and the measured
--                    masses. Also packages the full-register objective as a
--                    bounded self-adjoint operator (synOpFullL).
-- `Dqi.Witness`     NON-VACUITY: a concrete instantiation (l = 1, the [7,4]
--                    Hamming dual) discharging every hypothesis of
--                    dqi_end_to_end inside Lean, certifying the hypothesis
--                    cluster is jointly satisfiable.
import Dqi.Basic
import Dqi.Failure
import Dqi.Characters
import Dqi.LayerState
import Dqi.Objective
import Dqi.Reachable
import Dqi.Fourier
import Dqi.Amplitudes
import Dqi.SiteOp
import Dqi.Shell
import Dqi.Pullback
import Dqi.Residual
import Dqi.Channel
import Dqi.Embed
import Dqi.Flat
import Dqi.Counting
import Dqi.Degradation
import Dqi.Bridge
import Dqi.Boundary
import Dqi.ShellOp
import Dqi.Main
import Dqi.MainBoundary
import Dqi.FourierN
import Dqi.Phys
import Dqi.Intertwine
import Dqi.Taxonomy
import Dqi.PhysFull
import Dqi.Ewire
import Dqi.CrossPhys
import Dqi.Packing
import Dqi.Perturbation
import Dqi.Complete
import Dqi.Witness
