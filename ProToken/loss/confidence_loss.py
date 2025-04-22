import jax
import jax.numpy as jnp
import numpy as np

"""lDDT protein distance score."""
import jax.numpy as jnp

def softmax_cross_entropy(logits, label, seq_mask):
    # logits: (Nres, Nbins)
    # label: (Nres, Nbins) (one hot)
    # seq_mask: (Nres) 
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(log_probs * label, axis=-1)
    
    return jnp.sum(loss * seq_mask) / (jnp.sum(seq_mask) + 1e-6)

def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.

    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """

    assert len(predicted_points.shape) == 3
    assert predicted_points.shape[-1] == 3
    assert true_points_mask.shape[-1] == 1
    assert len(true_points_mask.shape) == 3

    # Compute true and predicted distance matrices.
    dmat_true = jnp.sqrt(1e-10 + jnp.sum(
        (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

    dmat_predicted = jnp.sqrt(1e-10 + jnp.sum(
        (predicted_points[:, :, None] -
        predicted_points[:, None, :])**2, axis=-1))

    dists_to_score = (
        (dmat_true < cutoff).astype(jnp.float32) * true_points_mask *
        jnp.transpose(true_points_mask, [0, 2, 1]) *
        (1. - jnp.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = jnp.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).astype(jnp.float32) +
                    (dist_l1 < 1.0).astype(jnp.float32) +
                    (dist_l1 < 2.0).astype(jnp.float32) +
                    (dist_l1 < 4.0).astype(jnp.float32))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + jnp.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + jnp.sum(dists_to_score * score, axis=reduce_axes))

    return score

class ConfidenceLoss():
    def __init__(self, config):
        self.config = config
        self.lddt_min = config["lddt_min"]
        self.lddt_max = config["lddt_max"]
        
        self.num_bins = config["num_bins"]
        self.bin_width = (self.lddt_max - self.lddt_min) / self.num_bins
        self.epsilon = 1e-5 ## @Liyh: add to config?
        
        self.loss_type = self.config.loss_type
        assert self.loss_type in ['softmax', 'integratedBCE'], \
                        'unsupported confidence loss type {}'.format(self.loss_type)
        
        self.bin_center = jnp.array(
                jnp.linspace(self.lddt_min + self.bin_width / 2, self.lddt_max - self.bin_width / 2, self.num_bins),
                jnp.float32
            )
        self.bin_edge_lower = jnp.array(
                jnp.linspace(self.lddt_min, self.lddt_max - self.bin_width, self.num_bins),
                jnp.float32
            )
        self.bin_edge_upper = jnp.array(
                jnp.linspace(self.lddt_min + self.bin_width, self.lddt_max, self.num_bins),
                jnp.float32
            )
        
        def span(lddt):
            ## [0, 40] -> 16, [40, 60] -> 8, [60, 70] -> 4, [70, 80] -> 2, [80, 100] -> 1
            # span = jnp.zeros_like(lddt)
            # span = jnp.where(lddt < 40, 16, span)
            # span = jnp.where((lddt >= 40) & (lddt < 60), 8, span)
            # span = jnp.where((lddt >= 60) & (lddt < 70), 4, span)
            # span = jnp.where((lddt >= 70) & (lddt < 80), 2, span)
            # span = jnp.where(lddt >= 80, 1, span)

            # @ZhangJ. [0, 50] -> 8, [50, 70] -> 4, [70, 80] -> 2, [80, 100] -> 1
            # 建议写入config中
            span = jnp.zeros_like(lddt)
            span = jnp.where(lddt < 50.0, 8.0, span)
            span = jnp.where((lddt >= 50.0) & (lddt < 70.0), 4.0, span)
            span = jnp.where((lddt >= 70.0) & (lddt < 80.0), 2.0, span)
            span = jnp.where(lddt >= 80.0, 1.0, span)
            
            return span
        
        self.get_span = span
        
        
        ### self.neighbors -> self.label_smoothing
        self.label_smoothing = self.config.label_smoothing
        
        if self.label_smoothing > 0.0:
            self.neighbors = self.config.neighbors
            neighbor_mask = np.ones((self.num_bins,self.num_bins))
            neighbor_mask = neighbor_mask - np.triu(neighbor_mask, self.neighbors) - np.tril(neighbor_mask, -self.neighbors)
            neighbor_mask = neighbor_mask / (np.sum(neighbor_mask,axis=-1,keepdims=True) + 1e-6)
            self.neighbor_mask = jnp.array(neighbor_mask, jnp.float32)
        
    def integrate_mask(self, lddt):
        # lddt: (Nres)
        span = self.get_span(lddt) # (Nres)
        
        lddt_lower_bound, lddt_upper_bound = lddt - span, lddt + span
        bin_mask = jnp.logical_and(
            (self.bin_center[None, :] >= lddt_lower_bound[:, None]), 
            (self.bin_center[None, :] <= lddt_upper_bound[:, None]))
        
        return bin_mask
    
    def integrated_bce_loss(self, plddt, lddt, seq_mask):
        # plddt: (Nres, Nbins), lddt: (Nres, ), seq_mask: (Nres)
        plddt_prob = jax.nn.softmax(plddt, axis=-1) # (Nres, Nbins)
        integrate_mask = self.integrate_mask(lddt) # (Nres, Nbins)
        integrated_prob = jnp.sum(plddt_prob * integrate_mask, axis=-1) # (Nres)
        
        # binary cross entropy
        return -jnp.sum(seq_mask * jnp.log(integrated_prob + self.epsilon)) / (jnp.sum(seq_mask) + self.epsilon)
    
    def softmax_cross_entropy_loss(self, plddt, lddt, seq_mask):
        # plddt: (Nres, Nbins), lddt: (Nres, ), seq_mask: (Nres)
        
        lddt = jnp.expand_dims(lddt, axis=-1)
        lddt_one_hot = jnp.logical_and(lddt >= self.bin_edge_lower[None, ...], 
                                       lddt < self.bin_edge_upper[None, ...]).astype(jnp.float32) # (Nres, Nbins)
        
        loss = softmax_cross_entropy(plddt, lddt_one_hot, seq_mask)
        if self.label_smoothing > 0.0:
            smoothed_labels = jnp.matmul(lddt_one_hot, self.neighbor_mask)
            smoothed_loss = softmax_cross_entropy(plddt, smoothed_labels, seq_mask)
            
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smoothed_loss
            
        return loss
        
    
    def __call__(self, plddt, lddt, seq_mask):
        if self.loss_type == 'softmax':
            return self.softmax_cross_entropy_loss(plddt, lddt, seq_mask)
        else:
            return self.integrated_bce_loss(plddt, lddt, seq_mask)