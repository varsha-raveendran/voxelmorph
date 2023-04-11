import pathlib
path = pathlib.Path('/vol/pluto/users/raveendr/code/voxelmorph/data/oasis')
subj_lst_m = [str(f/'slice_seg4.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
with open('labels_list.txt','w') as tfile:
    tfile.write('\n'.join(subj_lst_m))
