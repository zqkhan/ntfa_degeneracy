Visualization
=============

Visualization can be preformed immediatly after training with the current epoch or after training by loading an epoch with load_state() in the following lines:

.. code-block:: python
    
    import htfa_torch.dtfa as DTFA
    import htfa_torch.niidb as niidb
    import htfa_torch.tardb as tardb
    import htfa_torch.utils as utils
    
    tar_db_file = tardb.FmriTarDataset('path/to/file.tar')
    dtfa = DTFA.DeepTFA(spreng_db, num_factors=100, linear_params='None', query_name=".")

    checkpoint_files = glob.glob('*CHECK*dtfa*')
    state_name = max(checkpoint_files, key=os.path.getctime).split('.dtfa')[0]

    dtfa.load_state(state_name)

Once loaded, the user has a choice of running prebuilt visualizations such as the following:

.. code-block:: python
   
    dtfa.scatter_task_embedding(labeler=lambda x: x, figsize=None, colormap='Set1')
    dtfa.scatter_subject_embedding(labeler=lambda x: x, figsize=None, colormap='plasma')

Plotting the factor centers, and reconstruction is also prebuilt. The following code draws a random run and plots the centers, original brain, and reconstructions for each block:

.. code-block:: python

    subject_runs = tar_db.subject_runs()

    for i in np.random.choice(range(len(subject_runs)), 1):
        subject, run = list(subject_runs)[i]
        logging.info('Plotting factors for Participant %d, run %d' % (subject, run))
        blocks = [block for block in affvids_db.blocks.values() if block['subject'] == subject and block['run'] == run]
        blocks.sort(key=lambda b: b['times'][0])
        for block in blocks:
            index = block['id']
            dtfa.plot_factor_centers(index, labeler=block_task_labeler)
            dtfa.plot_original_brain(index, labeler=block_task_labeler)
            dtfa.plot_reconstruction(index, labeler=block_task_labeler)


The embeddings for the model may also be manually loaded like in the following function:

.. code-block:: python

    def fetch_embeddings(): 
        hyperparams = dtfa.variational.hyperparams.state_vardict()
        tasks = dtfa.tasks()
        subjects = dtfa.subjects()
        z_p_mu = hyperparams['subject_weight']['mu'].data
        z_s_mu = hyperparams['task']['mu'].data

        z_ps_mu, combinations = list(), list()
        for p in range(len(subjects)):
            sub_tasks = [b['task'] for b in tar_db.blocks.values() if b['subject'] == subjects[p]]
            combinations.append(np.vstack([np.repeat(subjects[p],len(sub_tasks)), np.array(sub_tasks)]))
            for t in range(len(sub_tasks)):
                task_index = [i for i, e in enumerate(tasks) if e == sub_tasks[t]]
                joint_embed = torch.cat((z_p_mu[p], z_s_mu[task_index[0]]), dim=-1)
                interaction_embed = dtfa.decoder.interaction_embedding(joint_embed).data
                z_ps_mu.append(interaction_embed.data.numpy())
        z_ps_mu = np.vstack(z_ps_mu)   
        combinations = np.hstack(combinations).T  

        # convert to dataframes
        z_p = pd.DataFrame(np.hstack([np.reshape(subjects, (len(subjects),1)), z_p_mu.numpy()]),
                           columns=['participant','x','y'])
        z_s = pd.DataFrame(np.hstack([np.reshape(tasks, (len(tasks),1)), z_s_mu.numpy()]),
                           columns=['stimulus','x','y'])
        z_ps = pd.DataFrame(np.hstack([combinations, z_ps_mu]),
                            columns=['participant','stimulus','x','y'])
        return z_p, z_s, z_ps

Further plots are up the user's discretion.

Additional Notes
----------------

[TODO]