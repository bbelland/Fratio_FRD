import numpy as np
from scipy.ndimage.measurements import center_of_mass 
from scipy.ndimage.interpolation import shift
import warnings
import matplotlib.pyplot as plt

################################################################################
##############################################################################################################

class FRDsolver(object):
    """FRD solver function for PSF inputs"""
    
    def __init__(self, FRDlist, knownimagelist, imagetosolve, varianceimage):
        """Generates a FRDsolver object"""

        self.FRDlist = FRDlist
        self.knownimagelist = knownimagelist
        self.imagetosolve = imagetosolve
        self.varianceimage = varianceimage
        
        self._checkFRDlist(FRDlist)
        self._checkimagelists(knownimagelist,imagetosolve)
        self._checklengths(FRDlist,knownimagelist)
        
        self.residuallist = np.array([np.nan])

        
    def _checkFRDlist(self,FRDlist):
        """Validates the input FRD list"""
        
        if not isinstance(FRDlist,np.ndarray):
            raise Exception('FRDlist should be a np.ndarray')
            
        if not np.array_equal(FRDlist,np.sort(FRDlist)):
            warnings.warn('FRDlist is not sorted!')  

    def _checkimagelists(self,knownimagelist,imagetosolve):
        """Validates the input images"""
        
        if not isinstance(knownimagelist,np.ndarray):
            raise Exception('knownimagelist should be a np.ndarray')
        
        if not isinstance(imagetosolve,np.ndarray):
            raise Exception('imagetosolve should be a np.ndarray')
            
        for image in knownimagelist:
            if np.shape(image) != np.shape(imagetosolve):
                raise Exception('The input images of known FRD should have the same ' +
                                'dimensions as the image to solve.') 

        #self._recenter_imagelist()

    def _recenter_image(self,image,centertoshiftto):
        """Shift an image to a given center, centertoshiftto"""
        return shift(image,np.array(centertoshiftto)-np.array(center_of_mass(image)))

    def _recenter_imagelist(self):
        """Shifts all images to the same center as the first image in the knownimagelist"""  

        center_to_shift = center_of_mass(self.knownimagelist[0])

        temporarylist = self.knownimagelist

        for element in range(len(temporarylist)):
             temporarylist[element] = self._recenter_image(temporarylist[element],center_to_shift)

        self.knownimagelist = temporarylist

        
    def _checklengths(self, FRDlist, knownimagelist):
        "Verifies that the FRD inputs match the inputted images"
        
        if not len(FRDlist) == len(knownimagelist):
            raise Exception('The length of FRDlist is not equal to the length of knownimagelist. ' + 
                            'Make sure each image has a corresponding FRD.')
            
        if len(FRDlist) == 0:
            raise Exception('At least one FRD must be input!')
        
        if len(FRDlist) == 1:
            warnings.warn('No meaningful result will occur when only one comparison FRD value is given.')

    def residual_calculate(self,imagetosolve,guessimage,varianceimage):
    
        residualval = 0
        currentimage = imagetosolve 
        modeltocompare = guessimage 
        varimage = varianceimage
        showplt = False
            
        centery, centerx = center_of_mass(currentimage) #Determine the center of the PSF. Ordered y, x as
        #it would appear in imshow() but most important thing is to be consistent with ordering
        centery = int(np.round(centery)) #Rounded to permit easier pixel selection.
        centerx = int(np.round(centerx))
        
        #print('The found values of centery, centerx are: ')
        #print(centery, centerx)
        
        if showplt: #Code generally should not output plots, but for debug/step-by-step analysis, the plot
            #is available. Note that showplt is a variable defined in this function, not an argument,
            #since it isn't intended to be generally used.
        
            plt.imshow(currentimage[(centery-3):(centery+3),(centerx-3):(centerx+3)])
            plt.title('Outer mask selection in the input (non-simulated) image')
            plt.show()
            
            plt.imshow(modeltocompare[(centery-3):(centery+3),(centerx-3):(centerx+3)])
            plt.title('Outer mask selection in the comparison (simulated) image')
            plt.show()
            
            plt.imshow(varimage[(centery-3):(centery+3),(centerx-3):(centerx+3)])
            plt.colorbar()
            plt.title('Variance image in model range')
            plt.show()
            
            plt.imshow(currentimage[(centery-3):(centery+3),(centerx-3):(centerx+3)]
                       - modeltocompare[(centery-3):(centery+3),(centerx-3):(centerx+3)],
                       cmap='bwr',vmax=500,vmin=-500)
            plt.colorbar()
            plt.title('Difference image between input and comparison')
            plt.show()        
            
            plt.imshow(np.divide(np.sqrt(2)*np.square(currentimage[(centery-3):(centery+3),(centerx-3):(centerx+3)]
                                                      - modeltocompare[(centery-3):(centery+3),(centerx-3):(centerx+3)]),
                                                     (varimage[(centery-3):(centery+3),(centerx-3):(centerx+3)])))
            plt.colorbar()
            plt.title('Chi square in each pixel')# is {}'.format())
            plt.show()
    
            
        #print('Added to residual val: ')
        
        residualval += np.sum(np.divide(np.square(currentimage[(centery-3):(centery+3),(centerx-3):(centerx+3)]
                                                  - modeltocompare[(centery-3):(centery+3),(centerx-3):(centerx+3)]),
                                        (varianceimage[(centery-3):(centery+3),(centerx-3):(centerx+3)])))*np.sqrt(2)
        
        residualval -= np.sum(np.divide(np.square(currentimage[(centery-1):(centery+1),(centerx-1):(centerx+1)] - 
                                                  modeltocompare[(centery-1):(centery+1),(centerx-1):(centerx+1)]), 
                                        (varianceimage[(centery-1):(centery+1),(centerx-1):(centerx+1)])))*np.sqrt(2)

        residualval = residualval/40 #Number of pixels in the calculation
        
        return residualval            
            
    def find_FRD_compare_positions(self, FRDlist, knownimagelist, imagetosolve,varianceimage):
        """MinFRD = find_FRD_compare_positions(self, FRDlist, knownimagelist, imagetosolve)
        Solves for minimal FRD and returns it.
        
        Parameters
        -----------
        
        FRDlist: list
            A list of FRDs of the corresponding PSF image arrays in imagelist
    
        knownimagelist: list
            A list of PSF image arrays of known FRDs    
    
        imagetosolve: array
            PSF image with unknown FRD.
        
        varianceimage: array
            Variance image corresponding to imagetosolve.

        """
        
        
        #Begin with validation of the inputs
        self._checkFRDlist(FRDlist)
        self._checkimagelists(knownimagelist,imagetosolve)
        self._checklengths(FRDlist, knownimagelist)
            
        residuallist = []
        minFRD = np.nan
    
        for FRDindex in range(len(FRDlist)):
            residual_current = self.residual_calculate(imagetosolve,knownimagelist[FRDindex],varianceimage)
            residuallist.append(self.residual_calculate(imagetosolve,knownimagelist[FRDindex],varianceimage))
            if residual_current == np.min(residuallist):
                minFRD = FRDlist[FRDindex] #Still need to return the metric
        
        if np.isnan(minFRD):
            raise Exception('No FRD residuals were calculated.')
            
        self.residuallist = residuallist
        self.minFRD = minFRD
        
        return (residuallist,minFRD)
    
    def returnFRDrange(self):
        if not any(np.isnan(self.residuallist)):
            raise Exception('Must run find_FRD_compare_positions first.')
        else:
            if np.sum([residuallist<1]) == 0:
                warnings.warn('No FRD value gives a chi squared under 1.')
            #Test to see if terms under 1 stay under 1?
            return (np.min(residuallist[residuallist<1]),np.max(residuallist[residuallist<1]))

