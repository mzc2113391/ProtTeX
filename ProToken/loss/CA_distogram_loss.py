### Implement of CA distogram loss, from mindspore ProToken code

import jax
import jax.numpy as jnp
import numpy as np
from common.config_load import Config
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

def softmax_cross_entropy(logits, labels, smooth_factor=0.0):
    # labels: one hot
    num_class = labels.shape[-1]
    loss_xe = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    loss_smooth = -jnp.sum( (jnp.ones_like(labels)/num_class) * jax.nn.log_softmax(logits, axis=-1), axis=-1)

    loss = (1.-smooth_factor)*loss_xe + smooth_factor*loss_smooth
    return jnp.asarray(loss)

def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    
    loss = -labels * log_p - (1. - labels) * log_not_p
    return jnp.asarray(loss)

class OrdinalXCE():
    def __init__(self, num_class, e=0.0, neighbors=1):
        self.num_class = num_class
        self.e = e # label_smoothing 

        self.neighbors = neighbors

        ### self.neighbors -> self.label_smoothing
        neighbor_mask = np.ones((self.num_class,self.num_class))
        neighbor_mask = neighbor_mask - np.triu(neighbor_mask, neighbors) - np.tril(neighbor_mask, -neighbors)
        neighbor_mask = neighbor_mask / (np.sum(neighbor_mask,axis=-1,keepdims=True) + 1e-6)
        self.neighbor_mask = jnp.array(neighbor_mask, jnp.float32)

    def __call__(self, prediction_logits, target_tensor):
        '''
        prediction_logits is the output tensor (without softmax) with shape [..., 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor, one-hot codes.
        '''
        input_shape = target_tensor.shape
        
        # (...):
        xent_loss = softmax_cross_entropy(logits=prediction_logits, labels=target_tensor)
        
        ### add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        # (None,bins):
        target_tensor_ = jnp.reshape(target_tensor, (-1, input_shape[-1]))
        smoothed_labels_ = jnp.matmul(target_tensor_, self.neighbor_mask)
        # (...,bins):
        smoothed_labels = jnp.reshape(smoothed_labels_, input_shape)

        # (...):
        smoothed_xent = softmax_cross_entropy(logits=prediction_logits, labels=smoothed_labels)
        
        final_loss = (1 - self.e)*xent_loss + self.e*smoothed_xent
        return final_loss

class BinaryFocalLoss():
    """Return Focal loss for Binary classifications.
    
    """
    def __init__(self, alpha=0.25, gamma=2., epsilon=1e-8, feed_in_logit=False, not_focal=False): ### Pass config.model.heads.X here
        self.alpha = alpha
        self.gamma = gamma
        self.feed_in_logit = feed_in_logit
        self.not_focal = not_focal
        self.epsilon = epsilon
        
    def _convert_logit(self, probs):
        # probs = mnp.clip(probs,1e-5,1.-1e-5)
        # probs = mnp.clip(probs, MS_SMALL, 1.-MS_SMALL)
        probs = jnp.clip(probs, self.epsilon, 1.-self.epsilon)
        logits = jnp.log(probs/(1-probs))
        return logits

    def __call__(self, logits, labels):
        '''
        logits is the output tensor with shape [None,] before sigmoid;
        labels is the label tensor, [None,], 0 or 1 valued.
        1: dist<15; 0: dist>15. (Compute the distribution of these two labels)
        '''
        # epsilon = self.epsilon
        
        labels = jnp.asarray(labels, jnp.float32)
        probs = jnp.asarray(logits, jnp.float32)
        if self.feed_in_logit:
            probs = jax.nn.sigmoid(logits)
        else:
            logits = self._convert_logit(logits)
        
        if self.not_focal:
            focal_loss = sigmoid_cross_entropy(logits, labels)
        else:
            # (None):
            _ones = jnp.ones_like(labels)
            # positive_pt = mnp.where(labels>1e-5, probs, _ones)
            # negative_pt = mnp.where(labels<1e-5, 1-probs, _ones)
            positive_pt = jnp.where(labels>0.5, probs, _ones)
            negative_pt = jnp.where(labels<0.5, 1-probs, _ones)
            
            # (None,):
            focal_loss = -self.alpha * jnp.power(1-positive_pt, self.gamma) * jnp.log(jnp.clip(positive_pt, self.epsilon, 1.)) - \
                (1-self.alpha) * jnp.power(1-negative_pt, self.gamma) * jnp.log(jnp.clip(negative_pt, self.epsilon, 1.))
            focal_loss *= 2.
        
        return focal_loss

class EstogramHead():
    def __init__(self, first_break, last_break, num_bins):
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins
        
        # for distogram only:
        self.breaks = jnp.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]
        
        # ->(Nbins):
        self.centers = self.breaks - 0.5*self.width ### Note there may be bugs in previous versions. @ZhangJ.
        
    def compute_dmat(self, positions):
        """Builds DistogramHead module.

        Arguments:
            distogram_logits: [N_res, N_res, N_bins].
        Returns:
          Dictionary containing:
            * probs: distance matrix, shape [N_res, N_res, N_bins].
            * dmat: array containing bin centers, shape [N_res, N_res].
        """
        ### positions: (Nres,3)
        dmat = jnp.square(jnp.expand_dims(positions, 1) - jnp.expand_dims(positions, 0))
        # (nres,nres):
        dmat = jnp.sqrt(jnp.sum(dmat,-1) + 1e-10)
        return dmat
        
    def compute_estogram(self, distogram_logits, decoy_distance_mat):
        # distogram_logits:(Nres,Nres,Nbins)
        # decoy_distance_mat: (Nres,Nres)
        # centers: (Nbins)
        output_shape = distogram_logits.shape

        # (N*N,Nbins):
        distogram_logits_ = jnp.reshape(distogram_logits, (-1, self.num_bins))
        # (N*N,):
        decoy_distance_mat_ = jnp.reshape(decoy_distance_mat, (-1,))

        # (1,Nbins):
        square_centers = jnp.reshape(self.centers,(1,-1))
        # (N*N,bins):
        square_centers_pad_ = jnp.broadcast_to(square_centers, distogram_logits_.shape)
        # (N*N,Nbins):
        estogram_ = jax.nn.softmax(distogram_logits_)
        # (1,Nbins)-(N*N,1) -> (N*N,Nbins)
        esto_centers_ = square_centers - jnp.expand_dims(decoy_distance_mat_,-1)

        # (N,N,bins):
        estogram = jnp.reshape(estogram_, output_shape)
        esto_centers = jnp.reshape(esto_centers_, output_shape)
        square_centers_pad = jnp.reshape(square_centers_pad_, output_shape)
        return estogram, esto_centers, square_centers_pad
    
    def _integrate(self, distogram_logits, integrate_masks):
        # distogram_logits:(Nres,Nres,Nbins); masks:(Nres,Nres,Nbins)
        probs = jax.nn.softmax(distogram_logits)
        integrate_masks = jnp.asarray(integrate_masks, jnp.float32)
        # (Nres,Nres):
        v = jnp.sum(probs*integrate_masks,-1)
        return v
    
    def __call__(self, distogram_logits, dist_gt, dist_mask, cutoff):
        ### distogram_logits: (Nres,Nres,bins); dist_gt:(Nres,Nres); dist_mask:(Nres,Nres); cutoff:float
        
        # # (Nres,Nres):
        # mask_2d = dist_mask
        # # (Nres,Nres):
        # pad_mask_2d = mask_2d.astype(jnp.float32)
        # # Exlude self-interactions
        # pad_mask_2d *= (1. - jnp.eye(pad_mask_2d.shape[0]))

        pad_mask_2d = dist_mask
        dmat_ref = dist_gt
        
        ### Compute Decoy Distance Matrix:
        # dmat_ref = self.compute_dmat(positions_gt)
        # dmat_ref = dmat_ref.astype(jnp.float32)
        # dmat_ref = dist_gt.astype(jnp.float32)
        
        lddt_mask = (dmat_ref<cutoff).astype(jnp.float32)
        lddt_mask *= pad_mask_2d

        # (Nres,Nres,Nbins), (Nres,Ners,Nbins):
        estogram, esto_centers, square_centers_pad = self.compute_estogram(distogram_logits, dmat_ref)
        # (Nres,Nres):
        pair_errors = jnp.sum(estogram*esto_centers,-1)

        # (Nres,Nres):
        p1 = self._integrate(distogram_logits, jnp.abs(esto_centers)<0.5).astype(jnp.float32)
        p2 = self._integrate(distogram_logits, jnp.abs(esto_centers)<1.0).astype(jnp.float32)
        p3 = self._integrate(distogram_logits, jnp.abs(esto_centers)<2.0).astype(jnp.float32)
        p4 = self._integrate(distogram_logits, jnp.abs(esto_centers)<4.0).astype(jnp.float32)

        # (Nres,Nres):
        p0 = self._integrate(distogram_logits, square_centers_pad<cutoff).astype(jnp.float32)
        # (Nres,Nres):
        pred_mask2d = p0 * pad_mask_2d
        
        ### @ZhangJ. changed below:
        # (Nres):
        norm = jnp.sum(lddt_mask,-1) + 1e-12
        p1 = jnp.sum(p1*lddt_mask,-1)
        p2 = jnp.sum(p2*lddt_mask,-1)
        p3 = jnp.sum(p3*lddt_mask,-1)
        p4 = jnp.sum(p4*lddt_mask,-1)

        # (Nres):
        plddt = 0.25*(p1+p2+p3+p4)/norm

        return plddt, pred_mask2d, pair_errors ### the first two terms will enter loss calculations.

class CA_DistogramLoss():
    def __init__(self, config):     
        ### Initialize   
        self.config = config
        self.dtype = jnp.float32
        self.num_bins = self.config["num_bins"]
        self.breaks = jnp.linspace(self.config["first_break"], self.config["last_break"], self.num_bins)
        
        self.contact_cutoff_min = self.config["contact_cutoff_min"] ### 8A
        self.contact_cutoff_max = self.config["contact_cutoff_max"] ### 15A
        
        self.label_smoothing = self.config["label_smoothing"] ### 0.1
        self.dgram_neighbors = self.config["dgram_neighbors"] # train_config.exclude_neighbors disto_mask zero intra-chain [i,i+3]

        self.dgram_regularizer = OrdinalXCE(num_class=self.num_bins, e=self.label_smoothing, neighbors=self.dgram_neighbors)
        self.label_permutations = self.config["label_permutations"]
        self.lddt_loss =  EstogramHead(self.config["first_break"], self.config["last_break"], self.num_bins)
        
        self.if_focal_loss = self.config["focal_loss"]
        self.focal_alpha = self.config["focal_alpha"]
        self.focal_gamma = self.config["focal_gamma"]
        if not self.if_focal_loss:
            self.focal_alpha = 0.5
            self.focal_gamma = 0.

        self.contact_loss = BinaryFocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, epsilon=1e-5,
                                            feed_in_logit=False, not_focal=bool(not self.if_focal_loss))
    
    def __call__(self, distogram_logits, dist_gt_perms, dist_mask_perms, perms_padding_mask, rng_key):
        #### Calculate 
        ### distogram_logits:(Nres,Nres,bins);
        ### dist_gt_perms:(P,Nres,Nres);
        ### dist_mask_perms:(P,Nres,Nres).
        ### perms_padding_mask:(P,)
        ### rng_key: ()

        # (P,Nres,Nres,bins):
        distogram_logits_flat = jnp.tile(jnp.expand_dims(distogram_logits, 0), (self.label_permutations,1,1,1))

        # ():
        random_cutoff_mask = jax.random.uniform(rng_key, dtype=self.dtype, minval=self.contact_cutoff_min, maxval=self.contact_cutoff_max) 
        # (P,1,1):
        random_cutoff_mask_flat = jnp.tile(jnp.reshape(random_cutoff_mask, (1,1,1)), (self.label_permutations,1,1))
        random_cutoff_mask_flat = jnp.asarray(dist_gt_perms < random_cutoff_mask_flat, jnp.float32)

        # (), (P):
        dmat_loss, opt_perm_mask = self.distogram_loss_func(distogram_logits_flat, dist_gt_perms, dist_mask_perms, perms_padding_mask, random_cutoff_mask_flat)

        # (Nres,Nres):
        # (P,Nres,Nres) * (P,1,1) -> (Nres,Nres)
        dist_gt_opt = jnp.sum(dist_gt_perms * jnp.reshape(opt_perm_mask, opt_perm_mask.shape+(1,1)), 0)
        # (Nres,Nres):
        dist_mask_opt = jnp.sum(dist_mask_perms * jnp.reshape(opt_perm_mask, opt_perm_mask.shape+(1,1)), 0)
        # (), ():
        lddt_loss, contact_loss = self.lddt_loss_func(distogram_logits, dist_gt_opt,
                                                      dist_mask_opt, random_cutoff_mask)

        return dmat_loss, lddt_loss, contact_loss
        
    def random_cutoff_mask(self, dist_mask_perms):
        """Log loss of a distogram."""
        # dist_mask_perms:(P,Nres,Nres);

        cutoff = jax.random.uniform(self.make_rng("uniform"), dist_mask_perms.shape[:1], dtype=self.dtype, minval=self.contact_cutoff_min, maxval=self.contact_cutoff_max) # (P,)
        cutoff = jnp.reshape(cutoff, (-1,1,1)) # (P,1,1)

        return cutoff
    
    def distogram_loss_func(self, logits, dist_gt_perms, dist_mask_perms, perms_padding_mask, cutoff_mask):
        """Log loss of a distogram."""
        # logits:(P,Nres,Nres,bins); dist_gt_perms:(P,Nres,Nres);
        # dist_mask_perms:(P,Nres,Nres); perms_padding_mask:(P).
        # cutoff_mask:(P,1,1)

        bin_edges = self.breaks #(bins,)
        # (P,Nres,Nres,1):
        dist = jnp.expand_dims(dist_gt_perms, -1)
        # (1,1,1,bins)
        breaks = jnp.reshape(bin_edges,(1,1,1,-1))
        # (P,Nres,Nres,bins):
        aa = (dist > breaks).astype(jnp.float32)

        # (P,Nres,Nres):
        true_bins = jnp.sum(aa, -1)
        true_bins = true_bins.astype(jnp.int32) # (B*P,Nres,Nres)
        # @ZhangJ. Avoid the last bin:
        true_bins = jnp.clip(true_bins, 0, self.num_bins-1)
        
        # (P,Nres,Nres,bins):
        labels = jax.nn.one_hot(true_bins, self.num_bins)

        # (P,Nres,Nres):
        errors = self.dgram_regularizer(prediction_logits=logits, target_tensor=labels)
        
        ### Compose relevant masks:
        # (P,Nres,Nres):
        square_mask = dist_mask_perms * cutoff_mask
        
        # We return an upper-triangle mat:
        # (P,Nres,Nres):
        loss_mask = jnp.triu(square_mask, k=0)
        
        # (P,):
        avg_error = (jnp.sum(errors * loss_mask, (-2, -1)) /
                     (1e-8 + jnp.sum(loss_mask.astype(jnp.float32), (-2, -1))))
        avg_error += (1. - jnp.asarray(perms_padding_mask, jnp.float32))*1e5

        # ():
        loss = jnp.mean(avg_error)
        optimal_index = jnp.argmin(avg_error)
        
        # (P,):
        optimal_mask = jax.nn.one_hot(optimal_index, self.label_permutations)

        return loss, optimal_mask
    
    def lddt_loss_func(self, distogram_logits, dist_gt, dist_mask, cutoff_mask):
        # (Nres,Nres,bins), (Nres,Nres), (Nres,Nres), ()

        plddt, pred_contact, pair_errors_ = self.lddt_loss(distogram_logits, dist_gt, dist_mask, cutoff=cutoff_mask)

        # (Nres):
        lddt_loss_mask = jnp.clip(jnp.sum(dist_mask, -1), 0., 1.)
        lddt_error = - jnp.log(jnp.clip(plddt, 1e-8, 1.0))
        # ():
        lddt_loss = jnp.sum(lddt_error*lddt_loss_mask) / (jnp.sum(lddt_loss_mask, -1) + 1e-8)

        # (Nres,Nres):
        contact_loss_mask = dist_mask
        # (Nres*Nres):
        contact_probs = jnp.reshape(jnp.asarray(pred_contact, jnp.float32), (-1))
        
        # (Nres,Nres):
        contact_labels = jnp.asarray(dist_gt<cutoff_mask, jnp.float32) * contact_loss_mask
        # (Nres*Nres):
        contact_labels = jnp.reshape(contact_labels, (-1))

        # (Nres*Nres):
        contact_error = self.contact_loss(logits=contact_probs, labels=contact_labels)
        # (Nres,Nres):
        contact_error = jnp.reshape(contact_error, contact_loss_mask.shape)
        # ():
        contact_loss = jnp.sum(contact_error*contact_loss_mask) / (jnp.sum(contact_loss_mask) + 1e-8)

        return lddt_loss, contact_loss