#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('cd', '../')


# In[2]:


import logging
import numpy as np
import htfa_torch.dtfa as DTFA
import htfa_torch.niidb as niidb
import htfa_torch.tardb as tardb
import htfa_torch.utils as utils


# In[3]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


# In[5]:


spreng_db = tardb.FmriTarDataset('data/mini_aging_site1_blocks.tar')


# In[6]:


dtfa = DTFA.DeepTFA(spreng_db, num_factors=100, embedding_dim=2)


# In[7]:


#dtfa.visualize_factor_embedding()


# In[8]:


#dtfa.num_blocks


# In[9]:


#dtfa.num_voxels


# In[10]:


losses = dtfa.train(num_steps=2000, learning_rate={'q': 1e-2, 'p': 1e-4}, log_level=logging.INFO, num_particles=1,
                    batch_size=256, use_cuda=True, checkpoint_steps=100, patience=50,)


# In[11]:


#utils.plot_losses(losses)


# In[12]:


dtfa.scatter_subject_weight_embedding(figsize=None, colormap='Set1', filename='spreng_blocks_subject_embedding.pdf')


# In[ ]:




