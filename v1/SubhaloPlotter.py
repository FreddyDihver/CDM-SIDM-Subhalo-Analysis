# Import necessary libraries and functions
import sys
import h5py
from SubFuncs import *
from SubPlotFuncs import *

##############
# User input #
##############
# Path to the directory containing the simulation data
simulation_directory = Path(directory(str(sys.argv[0]))).parent / 'Convergence Test' / 'Lvl 8 CDM'  # Change to appropriate path

# Paths to the FOF and snapshot files
fof = simulation_directory / 'fof_subhalo_tab_001.hdf5'     # Change to appropriate FOF file if needed
snap = simulation_directory / 'snap_001.hdf5'               # Change to appropriate snapshot file if needed

# Path to save the output figures, will be created if it does not exist
savefolder = simulation_directory / 'Figures'
savefolder.mkdir(parents=True, exist_ok=True)

# Constants and user input parameters
age = 10                # age of the system in Gyr for calculating the expected number of interactions if SIDM is selected
HubbleParam = 0.6774    # Dimensionless Hubble parameter for the simulation

isSIDM = int(input('Dark matter type CDM(0)/SIDM(1):\n'))
crosssect = float(input('Cross section in cm^2/g:\n')) if isSIDM else None
DataName = input('Data name for plot (e.g. "CDM" or "SIDM1"):\n')
BoxSize = int(input('Boxsize in Mpc:\n')) * 1000
particlemin = int(input('Minimum number of particles in subhalo to plot:\n'))
RelaxParam = int(input('Relaxation criteria off(0)/on(1):\n'))
Clean = int(input('Keep figures clean no(0)/yes(1):\n'))
Fiton = int(input('Make fits no(0)/yes(1):\n'))

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
    veldispall = np.array(snapfile['PartType1']['SubfindVelDisp'])      # velocity dispersion of all particles in the snapshot

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
        veldisp = veldispall[prev_i:prev_i + num_of_DMpart]

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

        ##############################
        # Plot particle distribution #
        ##############################
        # Plot the particle distribution of the subhalo, with optional markers for the center of mass 
        # and potential minimum, and a circle for the virial radius
        fig, ax = plot_subhalo(coord, CM[i], MaxPot[i], r200, Clean, c1, c2)

        # If not in clean mode, add a text box with information about the subhalo
        if not Clean:
            info = info_text(len(coord), sub_group[i], m200, virratio, relax)
            text_box(ax, info, x=0.45, y=0.2, color='r')
        # Save the figure for the particle distribution
        save_figure(fig, savefolder / f'subhalo{i}.jpg')

        # Calculate the power radius of the subhalo, and build the density profile from the particle data
        PowerRad = PowerRadius(r, pmass)
        r_bin_centers, dm_dens_mean = build_density_profile(r, dm_dens)

        ########################
        # Plot density profile #
        ########################
        plt.style.use('default')
        figdens = plt.figure(figsize=(5.5, 5))
        axdens = plt.axes([0.1, 0.1, 0.87, 0.87])

        # Check if the subhalo is a SIDM subhalo, and if so, fit the density profile with a composite profile
        if isSIDM and Fiton:
            # Fit the density profile with a NFW profile for the outer part, and plot the fit along with the data
            nfw_result, nfw_text = fit_nfw(r_bin_centers, dm_dens_mean, r200, PowerRad, Clean, HubbleParam, SIDMon=isSIDM)

            # If the NFW fit was successful, plot the NFW profile find the isothermal fit for the inner part of the profile
            if nfw_result is not None:
                centers, nfw_profile, _, popt_nfw = nfw_result

                # Calculate the radius where the expected number of interactions is 1 or more
                r1, _ = R1(crosssect, veldisp, age, r, popt_nfw)
                # Mask for the outer part of the profile where the NFW fit is valid, and plot the NFW profile 
                nfw_mask = centers > r1
                axdens.plot(centers, nfw_profile, 'g', alpha=0.5, label='NFW fit')
                axdens.plot(centers[nfw_mask], nfw_profile[nfw_mask],
                            'g', linewidth=4, linestyle='dashed')

                # Fit the density profile with an isothermal profile for the inner part
                iso_result, iso_text = fit_isothermal(r_bin_centers, dm_dens_mean, r1)
                # If the isothermal fit was successful, plot the isothermal profile
                if iso_result is not None:
                    centers, iso_profile, _ = iso_result
                    iso_mask = centers < r1
                    axdens.plot(centers, iso_profile, 'purple', alpha=0.5, label='Iso fit')
                    axdens.plot(centers[iso_mask], iso_profile[iso_mask],
                                'purple', linewidth=4, linestyle='dashed')
                    # Plot r1
                    r1plotdens = NFW(r1, popt_nfw[0], popt_nfw[1])
                    axdens.vlines(x = r1, ymin = 0.5*r1plotdens, ymax = 2*r1plotdens,
                                            color = 'orange', label = '$r_1$',linewidth=4,zorder=20)
                else:
                    iso_text = []

                # If not in clean mode, add a text box with information about the fits
                if not Clean:
                    text_box(axdens, '\n'.join(nfw_text + iso_text), x=0.67, y=0.98)

        # If the subhalo is not a SIDM subhalo, fit the density profile with a NFW profile and plot the fit along with the data
        elif not isSIDM:
            nfw_result, nfw_text = fit_nfw(r_bin_centers, dm_dens_mean, r200, PowerRad, Clean, HubbleParam)

            # If the NFW fit was successful, plot the NFW profile
            if nfw_result is not None:
                centers, nfw_profile, _, popt_nfw = nfw_result
                axdens.plot(centers, nfw_profile, 'g', linewidth=4,
                            linestyle='dashed', label='NFW fit')
                
                # If not in clean mode, add a text box with information about the NFW fit
                if not Clean:
                    text_box(axdens, '\n'.join(nfw_text), x=0.75, y=0.98, fontsize=10)

        # Plot the density profile data, and add labels, scales, and a vertical line for the power radius
        axdens.plot(r_bin_centers, dm_dens_mean, 'r', label=DataName)
        axdens.set_xlabel('Radius from center [kpc]', fontsize=12)
        axdens.set_ylabel(r'Density ($M_\odot kpc^{-3}$)', fontsize=12)
        axdens.set_xscale('log')
        axdens.set_yscale('log')
        axdens.set_ylim(min(dm_dens_mean) / 10)
        axdens.axvline(x=PowerRad, color='b', ls='--', label='Power Radius')
        axdens.legend(loc='lower left', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        ####################
        # Save and iterate #
        ####################
        # Save the figure for the density profile
        save_figure(figdens, savefolder / f'subhalo{i}_density.jpg')

        # Close the figures
        plt.close('all')

        # Iterate the previous index for the particle data to move to the next subhalo
        prev_i += num_of_DMpart