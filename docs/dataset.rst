Dataset
=======

The NTFA training loop takes in a tar db dataset file which is created with the webdataset utilities. To open a tar db file for writing, a user may call the following lines: 

.. code-block:: python

    import webdataset as wds
    sink = wds.TarWriter('to/your/file.tar')
 
Afterwards, Blocks are then used to indicate which TRs to include in the analysis and the relevant participants and conditions for the block. These are created using the FMriActivationBlock class. A user may create these in a number of ways, but current practice is to create a generator which yields a Block after parsing a csv file associated with a run. An example of it can be seen below:

.. code-block:: python

    class TaskElement:
        def __init__(self, task, start, end, run, fear_rating=None):
            def round_off_time(t):
                if t is not None:
                    if task != 'rest':
                        return round((t + TASK_ONSET_DELAY) / 0.8)
                    else:
                        return round(t)
                else:
                    return None
            self.task = task
            self.start_time = round_off_time(start)
            self.end_time = round_off_time(end)
            self.run = run
            self.fear_rating = fear_rating

    def parse_task_lines(lines):
        for (i, line) in enumerate(lines):
            cols = line.split(' ')
            task = cols[0]
            start_time = float(cols[5])
            end_time = float(cols[6])
            run = int(cols[4])
            fear_rating = float(cols[10])
            yield TaskElement(task, start_time, end_time, run, fear_rating)

    def rest_tasks(tasks):
        yield TaskElement('rest', 0, tasks[0].start_time - 1, tasks[0].run)
        for i in range(1, len(tasks)):
            rest_start = tasks[i-1].end_time + 1
            rest_end = tasks[i].start_time - 1
            if tasks[i].run == tasks[i-1].run:
                yield TaskElement('rest', rest_start, rest_end, tasks[i].run)
            else:
                yield TaskElement('rest', rest_start, None, tasks[i-1].run)
                yield TaskElement('rest', 0, rest_end, tasks[i].run)
        yield TaskElement('rest', tasks[-1].end_time + 1, None, tasks[-1].run)

    bad_runs = collections.defaultdict(lambda: [])
    bad_runs[3] = [1, 2, 3]
    bad_runs[7] = [3]
    bad_runs[14] = [3]
    bad_runs[22] = [1, 2, 3]
    bad_runs[23] = [1]
    bad_runs[24] = [1, 2, 3]
    bad_runs[26] = [3]

    def read_tasks(task_csv):
        def sentinel(f):
            return f if f is not None else 0.0
        with open(task_csv, 'r') as task_csv_file:
            subject = int(task_csv.split('_')[-1].strip('.txt'))
            logging.info('Subject %d', subject)
            task_lines = list(parse_task_lines(task_csv_file.readlines()))
            task_lines += list(rest_tasks(task_lines))
            rest_lines = [r for r in task_lines if r.task == 'rest']
            rest_lines = sorted(rest_lines, key=lambda t: sentinel(t.run))
            rest_starts_dict = {key: [] for key in range(1, 3)}
            rest_ends_dict = {key: [] for key in range(1, 3)}
            for (i,rest) in enumerate(rest_lines):
                if (rest.run in bad_runs[subject]) or (int(subject) not in include_subjects):
                    continue
                if rest.end_time is not None and rest.start_time is not None:
                    rest_ends_dict[rest.run].append(rest.end_time)
                    rest_starts_dict[rest.run].append(rest.start_time)
            task_lines = sorted(task_lines, key=lambda t: sentinel(t.start_time))
            for (i, task) in enumerate(task_lines):
                if task.run in bad_runs[subject] or (int(subject) not in include_subjects) or (task.task in exclude_tasks):
                    logging.info('Excluding block %d %s of run %d for motion', i,
                                 task.task, task.run)
                    continue
                logging.info('Block %d %s of run %d started at %f, ended at %f', i,
                             task.task, task.run, sentinel(task.start_time), sentinel(task.end_time))
                result = niidb.FMriActivationBlock(zscore=True, zscore_by_rest=True)
                result.subject = subject
                result.task = task.task
                result.run = task.run
                result.block = i
                result.start_time = task.start_time
                result.end_time = task.end_time
                result.rest_start_times = rest_starts_dict[result.run]
                result.rest_end_times = rest_ends_dict[result.run]
                result.individual_differences = {'fear_rating': task.fear_rating}
            yield result

Additional attributes may be added to the Block after yeilding from the generator. The Blocks are then written to the tar db file and a metadata file is also genearted from the total compiled trs and the final block's voxel locations. An example can be seen in the snippet below:


.. code-block:: python

    OVERRIDE = True

    if not os.path.exists(AFFVIDS_FILE) or OVERRIDE:
        total_trs = 0
        metadata = {
            'blocks': []
        }
        block_id = 0
        for task_csv in utils.sorted_glob(affvids_dir + task_log_csvs + '/*.txt'):
            for block in read_tasks(task_csv):
                block.filename = get_filename(block.subject, block.run)
                block.rest_end_times = '[' + ', '.join(map(str, block.rest_end_times)) + ']'
                block.rest_start_times = '[' + ', '.join(map(str, block.rest_start_times)) + ']'
                block.block = block_id
                block_id += 1
                block.mask = '/path/to/static/mask.nii.gz'
                block.smooth = 6
                block.load()
                metadata['blocks'].append(block.wds_metadata())

                for vals in block.format_wds():
                    sink.write(vals)
                block_trs = (block.end_time - block.start_time)
                total_trs += block_trs
        
        metadata['voxel_locations'] = block.locations
        metadata['num_times'] = total_trs
        torch.save(metadata, tar_file + '.meta')
        logging.info('Recorded metadata, including voxel locations')

    sink.close()

Further examples can be found in the notebooks directory.

Additional Notes
----------------

During the creation of the tar db file, ensure that the process is completed; if interupted, the user may have to close and reopen a TarWriter sink.