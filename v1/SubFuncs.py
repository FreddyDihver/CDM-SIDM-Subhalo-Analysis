import numpy as np
from scipy.optimize import curve_fit

##############################
# Subhalo analysis functions #
##############################
def RM200(r:list, pmass:float, rho_c = 127, h=0.6774)->float:
    """
    Calculates the virial radius and mass for overdensity Delta_c = 200,
    given radii in kpc, particle mass in Msol and critical density in Msol/kpc^3.
    Default rho_c = 127 Msol/kpc^3
    
    Returns r200 in kpc and M200 in 10^10 Msol.
    """
    # Check if empty list
    if len(r)==0:
        print('Empty list')
        return (None,None)
    
    r=np.sort(r)
    # Define radial bins
    maxr=r[-1]
    nbins = 100*int(maxr)
    
    # Find (cumulated) number in each bin
    n, edges = np.histogram(r, bins=nbins)
    Ncum = np.cumsum(n)
    rbin = edges[1:]
    
    # Search for r200
    i = 0
    while i<nbins:
        R = rbin[i]
        # Number of particles less than R
        N = Ncum[i]
        # Calculate volume inside R
        vol = 4/3*np.pi*R**3
        # Calculate mass and density
        M200 = N*pmass
        rho = M200/vol
        # If criteria satisfied -> r200 found
        if rho <= 200*rho_c/h**2:
            return (R, M200)
        i += 1

def R1(sigma_m, veldisp, age, r, NFW_params):
    """Calculates the radius where the expected number of interactions is 1 or more.

    Args:
        sigma_m (float): cross section in cm^2/g
        veldisp (array): Array of velocity dispersions
        age (float): Age of the system
        r (array): Array of radial distances
        NFW_params (tuple): Parameters for the NFW density profile

    Returns:
        tuple: Tuple containing the radius where the expected number of interactions is 1 or more and the expected number of interactions at that radius
    """
    # Define logarithmic bins for radius
    nbins = 200
    r_bins = np.logspace(np.log10(1), np.log10(r.max()), nbins)

    # Calculate the constant factor for the expected number of interactions
    const = sigma_m * age * 2.14e-10

    # Sort the radius and velocity dispersion arrays based on radius
    order = np.argsort(r)
    r_sorted = r[order]
    vel_sorted = veldisp[order]

    # Calculate the expected number of interactions at each radius
    for i in range(nbins - 1, 1, -1):
        # Create a mask for particles within the current radial bin
        r_low, r_high = r_bins[i - 1], r_bins[i]
        mask = (r_sorted > r_low) & (r_sorted < r_high)
        vellist = vel_sorted[mask]
        vellist = vellist[~np.isnan(vellist)]

        # Calculate the expected number of interactions using the NFW density profile and the average velocity in the bin
        if len(vellist):
            vel = 4 / np.sqrt(3 * np.pi) * np.mean(vellist)
            rho = NFW(r_bins[i], *NFW_params)
            # Calculate the expected number of interactions at this radius
            Nval = const * rho * vel 
            # Check if the expected number of interactions is 1 or more
            if Nval >= 1:
                return r_bins[i], Nval  # Return the radius and expected number of interactions

    return 0, 0


def build_density_profile(r, dm_dens, nbins=40):
    """Build density profile from r and dm_dens and return bin centers and mean densities in each bin.

    Args:
        r (array): Array of radial distances
        dm_dens (array): Array of dark matter densities
        nbins (int, optional): Number of bins for the density profile. Defaults to 40.

    Returns:
        tuple: Tuple containing the bin centers and the mean densities in each bin
    """
    # Define logarithmic bins for radius and calculate the mean density in each bin
    r_bins = np.logspace(np.log10(1), np.log10(max(r)), nbins)
    centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    means = np.array([
        dm_dens[(r >= r_bins[j]) & (r < r_bins[j + 1])].mean()
        for j in range(len(centers))
    ])
    # Return only the bins where the mean density is not NaN (meaning there are particles in that bin)
    valid = ~np.isnan(means)
    return centers[valid], means[valid]
def fit_nfw(r_bin_centers, dm_dens_mean, r200, PowerRad, clean, HubbleParam, SIDMon=False) -> tuple:
    """Fit a NFW profile to the density profile of a subhalo, and return the density profile and the parameters of the fit

    Args:
        r_bin_centers (ndarray): Array of radial bin centers
        dm_dens_mean (ndarray): Array of mean dark matter densities in each bin
        r200 (float): virial radius of the subhalo
        PowerRad (float): power radius of the subhalo
        clean (bool): if True, only plot the particle distribution without markers. Defaults to False.
        HubbleParam (float): Hubble parameter
        SIDMon (bool, optional): if True, apply an additional mask to exclude the inner part of the profile where the NFW fit is not valid for SIDM subhalos. Defaults to False.

    Returns:
        tuple: A tuple containing the fitted profile and fit information

    Returns:
        tuple: A tuple containing the fitted profile and fit information
    """
    # If clean mode is on, return None and an empty list for the fit information
    if clean:
        return None, []
    
    # If there are not enough bins within PowerRad to perform the fit, return None and an empty list for the fit information
    mask = (r_bin_centers > PowerRad) & (r_bin_centers < 0.5 * r200)
    if SIDMon:
        # For SIDM subhalos, also require that the bins are outside 10% of R200 to avoid the inner part where the NFW fit is not valid
        mask &= r_bin_centers > 0.1 * r200  
    if np.count_nonzero(mask) < 3:
        return None, []

    # Set the initial parameters for the NFW fit, and the bounds for the parameters to ensure a physically reasonable fit
    param = [np.mean(dm_dens_mean[:2]), 0.5 * r200]
    bound = [[0.01 * param[0], 0.01 * param[1]], [100 * param[0], 100 * param[1]]]

    # Try to fit the NFW profile
    try:
        # Fit the NFW profile to the density profile using curve_fit, and calculate the fitted profile
        popt, _ = curve_fit(NFW, r_bin_centers[mask], dm_dens_mean[mask],
                            param, bounds=bound, sigma=np.ones(np.count_nonzero(mask)))
        profile = NFW(r_bin_centers, *popt)

        # Calculate the R-squared value for the fit in log-log space to assess the quality of the fit
        Rsqlog = Rsquared(np.log(dm_dens_mean), np.log(profile))

        # Return the fitted profile and fit information
        info = [
            fr'NFW',
            fr' $R^2_{{log}}$ = {round(Rsqlog, 3)}',
            fr'  $\rho_0$ = {popt[0]:.1E}',
            fr'  $R_s$ = {popt[1]:.1E}'
        ]
        return (r_bin_centers, profile, mask, popt), info
    # If the fit fails, return None and an empty list for the fit information
    except (ValueError, ZeroDivisionError):
        return None, []


def fit_isothermal(r_bin_centers, dm_dens_mean, r1) -> tuple:
    """Fit an isothermal profile to the density profile of a subhalo, and return the density profile and the parameters of the fit

    Args:
        r_bin_centers (ndarray): Array of radial bin centers
        dm_dens_mean (ndarray): Array of mean dark matter densities in each bin
        r1 (float): Radius for the isothermal fit

    Returns:
        tuple: A tuple containing the fitted profile and fit information
    """
    #If there are not enough bins within r1 to perform the fit, return None and an empty list for the fit information
    mask = r_bin_centers < r1
    if np.count_nonzero(mask) < 3:
        return None, []

    # Set the initial parameters for the isothermal fit, and the sigma for the curve fit to give more weight to the inner bins
    param = [np.mean(dm_dens_mean[:2]), 100]
    sigma = np.ones(np.count_nonzero(mask))
    sigma[-1] = 0.005

    # Try to fit the isothermal profile
    try:
        # Use curve_fit to fit the isothermal profile
        popt, _ = curve_fit(isothermal, r_bin_centers[mask],
                            dm_dens_mean[mask], param, sigma=sigma)
        profile = isothermal(r_bin_centers, *popt)

        # Calculate the R-squared value for the fit in log-log space to assess the quality of the fit
        Rsqlog = Rsquared(np.log(dm_dens_mean[mask]), np.log(profile[mask]))
        # Return the fitted profile and fit information
        info = [
            fr'Iso',
            fr' $R^2_{{log}}$ = {round(Rsqlog, 3)}',
            fr' $\rho_0$ = {popt[0]:.1E}',
            fr' $R_0$ = {popt[1]:.1E}'
        ]
        return (r_bin_centers, profile, mask), info
    
    # If the fit fails, return None and an empty list for the fit information
    except (ValueError, ZeroDivisionError):
        return None, []
    
def directory(filepath:str):
    i=filepath.rfind('/')
    return filepath[:i+1]

def dist_from_center(center:np.array, coord:np.array, Boxsize:int):
    """
    Calculates distance from particle with coord to center of halo
    using periodic boundary conditions. Boxsize in kpc
    """
    x,y,z=coord[:]
    cx,cy,cz=center[:]
    dx = min(abs(cx-x), Boxsize-abs(cx-x))
    dy = min(abs(cy-y), Boxsize-abs(cy-y))
    dz = min(abs(cz-z), Boxsize-abs(cz-z))
    return np.sqrt(dx*dx+dy*dy+dz*dz)

def NFW(r:float, rho0:float, Rs:float)->float:
    """
    Returns the NFW profile for central density rho0, scale radius Rs and
    radius r.
    """
    rho0=abs(rho0)
    Rs=abs(Rs)
    r_Rs=r/Rs
    #return rho0*Rs*Rs/(r*(Rs+2*r+r*r))
    return rho0/(r_Rs*(1+r_Rs)**2)

def exponential(r:float, rho0:float, Rs:float)->float:
    """
    Returns the exponential profile for central density rho0, scale radius Rs
    and radius r.
    """
    rho0=abs(rho0)
    Rs=abs(Rs)
    return rho0*np.exp(-r/Rs)

def isothermal(r:float, rho0:float, r0:float)->float:
    """
    Returns the isothermal profile for rh.
    """
    rho0=abs(rho0)
    return rho0/(1+r*r/(r0*r0))

def PowerRadius(r:list, pmass:float, rho_c = 127, h = 0.6774,Nup = 1)->float:
    """
    Returns the Power radius for a group of particles with radial distance r
    from the point of max potential, given H0=67.74.
    That is, the smallest radius satisfying:
    sqrt(200)/8 * N/ln(N) * sqrt(rhoc/mean(rho)
    Default: rho_c = 127 Msol/kpc^3.
    Nup is the upscale amount for testing
    """
    # Check if empty list
    if len(r)==0:
        print('Empty string r')
        return None

    r=np.sort(r)
    #Define radial bins
    maxr=r[-1]
    nbins = 10*int(maxr)
    # Find (cumulated) number in each bin
    n, edges = np.histogram(r, bins=nbins)
    Ncum = np.cumsum(n)
    rbin = edges[1:]

    #Search for Power radius
    i = 0
    while i<nbins:
        R = rbin[i]
        # Number of particles less than R
        N = Ncum[i]*Nup
        # Calculate mean density 
        rhomean = pmass*N/(rbin[i]**3)
        # If criterion satisfied -> Power radius found
        crit = 1.77*N/np.log(N)*np.sqrt(rho_c/(h**2*rhomean)) 
        if crit >= 1 and N > 1:
            return rbin[i]
        i += 1

def Rsquared(obs:list,exp:list)->float:
    """
    Returns the Rsquared value for observed values obs and expected values exp.
    """
    meanobs = np.mean(obs)
    SSres = sum((obs-exp)**2)
    SStot = sum((obs-meanobs)**2)
    return 1 - SSres/SStot