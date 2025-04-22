"""Training config"""
import ml_collections
import copy

TRAINING_CONFIG = {'seq_len_power': 0.5,
                   'weight_decay': 1e-5,
                   
                   'neighbor': {'cutoff': 8.0},
                   
                   'fape': {'atom_clamp_min_distance': 0.3, 
                            'atom_clamp_distance': 10.0,
                            'loss_unit_distance': 10.0,
                            'loss_weight': 1.0, # 0.5 is recommended by AF2
                            'IPA_weight': [0.125, 0.14285715, 0.16666667, 0.2, 0.25, 0.33333334, 0.5, 1.0], # c.f. MAML++; w=1/(T-t+1)
                            'clamp_prob': 0.9,
                            'adversarial_scaling': 0.01
                            },
                   
                   'structural_violation': {'clash_overlap_tolerance': 1.5,
                                            'violation_tolerance_factor': 12.0,
                                            'loss_weight': 0.0, # 0.06
                                            },
                   
                   'distogram':{'first_break': 3.0,
                                'last_break': 20.5,
                                'num_bins': 36,
                                'label_smoothing': 0.1,
                                'label_permutations': 1, # @ZhangJ. 注意，该超参应该移入FAPE
                                'dgram_neighbors': 1, ### 2 -> 1
                                'contact_cutoff_min': 8.0, # unit:A
                                'contact_cutoff_max': 15.0, # unit:A
                                
                                'focal_loss': False, # True
                                'focal_alpha': 0.5, # need double check @Zhenyu. >0.9 is OK?
                                'focal_gamma': 2, # need double check @Zhenyu.
                                        
                                'weight': 0.2, # 0.3 is recommended by AF2
                                'w1': 0.0, # distogram
                                'w2': 5.0, # contact
                                'w3': 0.5, # lddt
                                },
                   
                    'inverse_folding': {'loss_weight': 0.0,
                                        'generator_loss_weight': 0.0, 
                                        'critic_loss_weight': 4.0
                                        },
                    
                    'confidence': {'lddt_min': 0,
                                    'lddt_max': 100,
                                    'num_bins': 50, # @ZhangJ. 检查np.range(0,100,50)的输出
                                    'loss_weight': 0.0, # 0.01
                                    'loss_type': 'softmax', ### 'integratedBCE'
                                    'label_smoothing': 0.1,
                                    'neighbors': 2, 
                                    },
                    
                    'mutual_information': {
                        'stop_code_grad_in_post_loss': True, 
                        'post_1_loss_weight': 1.0, 
                        'post_2_loss_weight': 0.0, 
                        'gamma': 1.0,
                        'label_smoothing': 0.1,
                        'tau': 0.07,
                        'fix_tau': False, # True,
                        'tau_upper_bound': 0.2,
                        'tau_lower_bound': 0.01,
                        'tau_scaling_factor': 2.0
                    },
                    
                    'vq': {
                        'gumbel_grad_ratio': 0.0,
                        'e_latent_loss_weight': 2.5,
                        'q_latent_loss_weight': 22.5, 
                        'entropy_loss_weight': 1.0
                    },

                    'uniformity': {'loss_weight': 1.0,
                                    'temperature': 2.0},
                    
                    'code_consistency': {'period': 5000, 
                                         'decay_time_scale': 100000,
                                         'loss_weight_min': 8.0, #10.0,
                                         'loss_weight_max': 12.0, # 20.0, 
                                         # 'loss_weight': 10.0,
                                         'loss_weight_gamma': 0.5,
                                         'lddt_threshold': 90.0, # 0.90,
                                         'tmscore_threshold': 0.90,
                                         'adversarial_grad_ratio': 0.5,
                                         'infoNCE_temperatures': [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
                                         'infoNCE_loss_weight': 0.02,
                                         },
                    
                    'lr': {'lr_max': 2e-4,
                           'lr_min': 2e-5,
                           'lr_init': 1e-6,
                           'warmup_steps': 10000,
                           'start_step': 0,
                           'lr_decay_steps': 80000,
                          },
                    }

TRAINING_CONFIG = ml_collections.ConfigDict(copy.deepcopy(TRAINING_CONFIG))