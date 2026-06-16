# Import necessary libraries and functions
import sys
import h5py
from SubFuncs import *
from SubPlotFuncs import *

############## 
# User input #
##############
# Path to the directory containing the simulation data
simulation_directory = Path(directory(str(sys.argv[0]))).parent / 'Path' / 'To' / 'Simulation' / 'Data' # Change to appropriate path

# Snapshot number/file ending
file_end = '001'

# Paths to the FOF and snapshot files
fof = simulation_directory / f'fof_subhalo_tab_{file_end}.hdf5'
snap = simulation_directory / f'snap_{file_end}.hdf5'

# Path to save the output figures, will be created if it does not exist
savefolder = simulation_directory / 'Subhalo Color Plots'
savefolder.mkdir(parents=True, exist_ok=True)

# Constants and user input parameters
HubbleParam = 0.6774    # Dimensionless Hubble parameter for the simulation
BoxSize = int(input('Boxsize in Mpc:\n')) * 1000
particlemin = int(input('Minimum number of particles in subhalo to plot:\n'))
RelaxParam = int(input('Relaxation criteria off(0)/on(1):\n'))
isSIDM = int(input('Dark matter type CDM(0)/SIDM(1):\n'))
crosssect = float(input('Cross section in cm^2/g:\n')) if isSIDM else None
DataName = input('Data name for plot (e.g. "CDM" or "SIDM1"):\n')
Clean = int(input('Keep figures clean no(0)/yes(1):\n'))

# Coordinates for 2D plots, 0,1,2 for x,y,z
c1, c2 = 0, 1

###################
# Data processing # 
###################
with h5py.File(fof, 'r') as foffile, h5py.File(snap, 'r') as snapfile:
    # Load data from the FOF and snapshot files
    size = np.array(foffile['Subhalo']['SubhaloLenType'])               # number of particles of each type in each subhalo
    CM = np.array(foffile['Subhalo']['SubhaloCM'])                      # center of mass position of each subhalo
    MaxPot = np.array(foffile['Subhalo']['SubhaloPos'])                 # position of the particle with the extremum potential in each subhalo
    coordsall = np.array(snapfile['PartType1']['Coordinates'])          # coordinates of all dark matter particles in the snapshot
    dm_densall = np.array(snapfile['PartType1']['SubfindDMDensity'])    # dark matter density of all particles in the snapshot

    # Extract group information from the FOF file
    Groupsize = np.array(foffile['Group']['GroupLenType'])[:, 1]
    firstsub = np.array(foffile['Group']['GroupFirstSub'])
    hassub = [e != -1 for e in firstsub]
    sub_group = np.array(foffile['Subhalo']['SubhaloGrNr'])

    # Calculate the particle mass from the FOF file
    pmass = (foffile['Group']['GroupMassType'][0, 1] / Groupsize[0]) * 1e10
    print('Particle mass: ', '{:.5E}'.format(pmass), 'Msol')

    # Initialize variables
    Group_i = 0
    prev_i = 0
    n_run = len(size)

    ##########################
    # Loop over all subhalos #
    ##########################
    for i in range(n_run):
        # If a subhalo belongs to a group with fewer particles than the specified minimum,
        # break the loop (since subhalos are ordered by group size)
        if Groupsize[Group_i] < particlemin:
            break

        # Get the number of dark matter particles in the current subhalo
        num_of_DMpart = size[i, 1]

        # If the current subhalo belongs to a different group than the previous one,
        # update the group index and the starting index for the particle data
        if sub_group[i] > Group_i:
            Group_i += 1
            prev_i = sum(Groupsize[:Group_i])
        if not hassub[Group_i]:
            Group_i += 1
            prev_i += sum(Groupsize[:Group_i])

        # Get the coordinates and dark matter density for the current subhalo
        coord = coordsall[prev_i:prev_i + num_of_DMpart]
        dm_dens = dm_densall[prev_i:prev_i + num_of_DMpart] * 1e10

        # If the number of particles in the subhalo is less than the specified minimum, skip this subhalo and move to the next one
        if len(coord) <= particlemin:
            prev_i += num_of_DMpart
            continue

        # Calculate the distance of each particle from the potential minimum of the subhalo, taking into account periodic boundary conditions
        r = np.array([dist_from_center(MaxPot[i], e, BoxSize) for e in coord])

        # Calculate the centralicity, to ensure that the potential minimum is not too close to the edges of the box
        # (to avoid issues with periodic boundary conditions)
        minc = min(0.08, 20000 / BoxSize)
        maxc = max(0.92, 1 - 20000 / BoxSize)
        centertest = np.array([MaxPot[i][j] / BoxSize > minc and MaxPot[i][j] / BoxSize < maxc
                               for j in range(3)])

        # Calculate the virial radius and mass of the subhalo, and the relaxation parameter
        try:
            r200, m200 = RM200(r, pmass)
            relax = dist_from_center(CM[i], MaxPot[i], BoxSize) / r200
        except TypeError:
            r200, m200 = None, None
            relax = 1
        relaxpar = 0.07

        # Calculate the virial ratio of the subhalo, if kinetic and potential energy data are available in the FOF file
        try:
            kin = foffile['Subhalo']['SubhaloEkin'][i]
            pot = foffile['Subhalo']['SubhaloEpot'][i]
            virratio = 2 * kin / abs(pot)
        except KeyError:
            print('No kinetic and potential energy in files!')
            virratio = 1

        # Set a threshold for the virial ratio to ensure that the subhalo is not too far from virial equilibrium
        virlim = 1.35

        # If the relaxation parameter is above the specified threshold, the potential minimum is too close to the edges of the box,
        # or if the virial ratio is above the specified limit, skip this subhalo and move to the next one
        if RelaxParam and not (np.all(centertest) and relax < relaxpar and virratio < virlim):
            prev_i += num_of_DMpart
            continue

        print(i)

        ##########################
        # Plot density colormesh #
        ##########################
        fig, ax = plot_subhalo_colormesh(coord, CM[i], MaxPot[i], pmass, r200, Clean, c1, c2)

        # If not in clean mode, add a text box with information about the subhalo
        if not Clean:
            info = info_text(len(coord), sub_group[i], m200, virratio, relax)
            text_box(ax, info, x=0.60, y=0.98)

        ####################
        # Save and iterate #
        ####################
        save_figure(fig, savefolder / f'subhalo{i}_colormesh.jpg')
        plt.close('all')

        # Iterate the previous index for the particle data to move to the next subhalo
        prev_i += num_of_DMpart