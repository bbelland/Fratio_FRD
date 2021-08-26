import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass #* Check if required
from scipy import interpolate #* Check if required
from scipy.ndimage.interpolation import shift#* Check if required
import warnings
import matplotlib.pyplot as plt
import os
import pickle
from FRDSolverClass import *

################################################################################
##############################################################################################################


class FRD_finder(object):
    """
    Class to find input unknown FRD images, compare them to a input/generated image set of known FRD,
    and plot them in a meaningful way."""
    
    def __init__(self,element,fratio,date, FRDlist, knownimagelist, imagetosolve, vartosolve,
                 exposurenumber,positionlist,base_DIRECTORY):
        """
        element = 'Ar', 'Kr', or 'Ne' (But can use any file with similar name
        format).
        
        fratio is used to read files from a specific folder. Can be replaced
        to more conveniently read in file names in other systems.
        
        date is used to read the proper wavefront solutions from files.
        date is of the form 'Aug0121'.
        
        FRDlist is a list of each FRD used in the images to compare with an
        image with unknown FRD.
        
        knownimagelist is the list of images with the known FRDs corresponding
        to FRD list. Can pass None to generate this automatically
        
        imagetosolve is the image (matrix) with unknown FRD. Can pass None to
        read in the files.
        
        vartosolve is the variance corresponding to imagetosolve, of the
        same size. Can pass None to read in the files.
        
        exposurenumber simply is used to identify the proper files from a folder
        containing the images.
        
        positionlist contains the list of all positions considered. Used to generate
        FRD images if images are not supplied, else, its length is used to iterate
        over each image solved
        """
        
        self.element = element
        self.fratio = fratio
        self.date = date
        self.FRDlist = FRDlist
        
        self.fratioFilename = str(int(np.floor(self.fratio))) + '_' + str(int(np.floor((10*self.fratio)%10)))
                             #Format for reading folder containing images
      
        self.exposurenumber = exposurenumber
        
        self.base_DIRECTORY = base_DIRECTORY
                    
        if positionlist is not None:
            self.positionlist = positionlist
            self.validpositions = positionlist
            
        else:
            self.positionlist = self.detect_files()
            self.validpositions = self.detect_files()
            
        if (imagetosolve is not None) and (vartosolve is not None):
            self.imagetosolve = imagetosolve
            self.vartosolve = vartosolve
        
        elif (imagetosolve is not None):# and vartosolve is None, consequently
            self.imagetosolve = imagetosolve
            self.vartosolve = imagetosolve #Approximate to first order            
        
        else:
            (self.imagetosolve,self.vartosolve) = self.generate_imagetosolve()
    
        if knownimagelist is not None:
            self.knownimagelist = knownimagelist
            
        else:
            self.knownimagelist = self.generate_FRDimages()     

        #Change directory to match folder
        
        self._test_types()
  
        self.residuallisttotal = None
        self.minFRDtotal = None
        self.FRD2dresiduals = None
        

    def _test_types(self):
        if not isinstance(self.element,str):
            raise Exception('element should be a string (e.g. Argon is "Ar").')
        if not isinstance(self.fratio,float):
            raise Exception('fratio should be a float (focal ratio of observation).') 
        if not isinstance(self.date,str):
            raise Exception('date should be a string (e.g. Aug0121, for the calibration file reading).')    
        if not isinstance(self.FRDlist,list) and not isinstance(self.FRDlist,np.ndarray):
            raise Exception('FRDlist should be a list or numpy array.')
        if not isinstance(self.knownimagelist,list) and not isinstance(self.knownimagelist,np.ndarray):
            raise Exception('knownimagelist should be a list or numpy array.')
        if not isinstance(self.imagetosolve,list) and not isinstance(self.imagetosolve,np.ndarray):
            raise Exception('imagetosolve should be a list or numpy array.') 
        if not isinstance(self.vartosolve,list) and not isinstance(self.vartosolve,np.ndarray):
            raise Exception('vartosolve should be a list or numpy array.') 
        if not isinstance(self.exposurenumber,str) and not isinstance(self.exposurenumber,int):
            raise Exception('exposurenumber should be a string or integer.')    
        if not isinstance(self.positionlist,list) and not isinstance(self.positionlist,np.ndarray):
            raise Exception('positionlist should be a list or numpy array.') 
        if not isinstance(self.base_DIRECTORY,str):
            raise Exception('base_DIRECTORY should be a string.')    
            
    def detect_files(self):
        """
        Finds valid filenames for a given element and focal ratio and outputs
        those filenames as a list, for use in the class.
        """
        


        positionsElementFratio = []
        for filenames in os.listdir(self.base_DIRECTORY + 'Samplef_' + self.fratioFilename + '/'):
                             #Read from separate image file folder
            positionval = filenames[8:-14] #Characters in filename corresponding to position on the detector
            elementfile = filenames[-14:-12] #Characters in filename corresponding to element
            if elementfile == self.element: #Check the file's element matches the tested element
                positionsElementFratio.append(int(positionval)) #Adds the corresponding position to a list

        positionsElementFratio = np.sort(list(set(positionsElementFratio))) #Organize list                          
                             
        with open(self.base_DIRECTORY+'results_of_fit_many_direct_' + self.element
                  + '_from_' + self.date + '.pkl','rb') as f:
                Results_Interpolation = pickle.load(f)
        
        positionsvalidElement = Results_Interpolation['0'].index.tolist() # A list of positions in the
                             #wavefront files, e.g. the list of valid positions to test.
        positionsvalidElement = list(map(int, positionsvalidElement)) #Simply converts the list of indices in
                             #Results_Interpolation into integers for comparison with positionsAr25


        positionsElementFratio = list(set(positionsElementFratio) & set(positionsvalidElement)) #If position
                             #in data file and position in the wavefront file, include it in the list to test
                             
        return positionsElementFratio
                             
    def generate_imagetosolve(self):
        imagetosolve = []        
        vartosolve = []                           
        for imageposition in self.positionlist:
            imagetosolve.append(np.load(self.base_DIRECTORY+'Samplef_'
                + self.fratioFilename + '/sci' + str(self.exposurenumber)
                + str(imageposition) + self.element + '_Stacked.npy'))
            vartosolve.append(np.load(self.base_DIRECTORY+'Samplef_'
                + self.fratioFilename + '/var' + str(self.exposurenumber)
                + str(imageposition) + self.element + '_Stacked.npy'))
        return (np.array(imagetosolve),np.array(vartosolve))
                             
    def generate_FRDimages(self):
        """
        Generate FRDimages for each position tested, if no images are presupplied.
        """
    
        residuallisttotalFratio = {}
        minFRDtotalFratio = {}
        FRD2dresidualsFratio = {}
                            
        datefinal = 'Jul2021' #datefinal corresponds to the date of the overview file, not of the file
                              # with wavelength solutions.

        for position in self.validpositions:
            #print(position)
            unknown_Fratio = self.imagetosolve
                             
            varimage_Fratio = self.vartosolve
        
            FRDimagesetFratio = {
                "FRDtofind" : {},
                "knownFRDlist" : {}
            } #Dictionary of the image to solve for and images for all known FRD value images.

            FRDimagesetFratio["FRDtofind"] = unknown_Fratio
            FRDimagesetFratio["knownFRDlist"] = generate_image_of_known_FRD_vartrue(unknown_Fratio,
                self.FRDlist,position,varimage_Fratio,self.element,self.base_DIRECTORY,datefinal)

            imagelistFratio = np.array(FRDimagesetFratio["knownFRDlist"])

            #imagetosolveFratio = FRDimagesetFratio["FRDtofind"]
                             
        return (imagelistFratio)

    def solve_FRD(self):
        residuallisttotal = {}
        minFRDtotal = {}
        FRD2dresiduals = {}
        
        for positionindex in range(len(self.validpositions)):
            
            #print(positionindex)
            #print(self.imagetosolve)
            #print(self.imagetosolve[positionindex])
            testFRDFratio = FRDsolver(self.FRDlist, self.knownimagelist,
                                      self.imagetosolve[positionindex],self.vartosolve[positionindex])

            (residuallistFratio,minFRDFratio) = testFRDFratio.find_FRD_compare_positions(self.FRDlist, 
                self.knownimagelist, self.imagetosolve[positionindex],self.vartosolve[positionindex])

            residuallisttotal[str(self.validpositions[positionindex])] = residuallistFratio
            minFRDtotal[str(self.validpositions[positionindex])] = minFRDFratio
            FRD2dresiduals[str(self.validpositions[positionindex])] = \
                           self.knownimagelist[self.FRDlist == minFRDFratio] - self.imagetosolve[positionindex]
            FRD2dresiduals[str(self.validpositions[positionindex])] = \
                                    FRD2dresiduals[str(self.validpositions[positionindex])][0] #Remove type info
        
        self.residuallisttotal = residuallisttotal
        self.minFRDtotal = minFRDtotal
        self.FRD2dresiduals = FRD2dresiduals
        
        return (residuallisttotal,minFRDtotal,FRD2dresiduals)

    def plot_results_one(self,positionindex):
        
        plt.plot(self.FRDlist,self.residuallisttotal[positionindex])
        plt.title(positionindex)
        plt.xlabel('FRD value')
        plt.ylabel('Sum of squares of residuals in mask')
        plt.show()

        maxvalue = (np.max(self.FRD2dresiduals[positionindex]))
        absminvalue = np.max(np.multiply(-1,(self.FRD2dresiduals[positionindex])))
        absmax = max(maxvalue,absminvalue)
        plt.imshow(self.FRD2dresiduals[positionindex],vmin=-absmax,vmax=absmax,origin='lower')
        plt.colorbar()
        plt.title('Position: {}'.format(positionindex))
        plt.show()    
    
    def plot_results_all(self):
        
        for position in range(len(self.residuallisttotal)):
            positionindex = str(self.positionlist[position])
            plt.plot(self.FRDlist,self.residuallisttotal[positionindex])
            plt.title(positionindex)
            plt.xlabel('FRD value')
            plt.ylabel('Sum of squares of residuals in mask')
            plt.show()
        
        #print(len(self.positionlist))
        for position in range(len(self.positionlist)):
            positionindex = str(self.positionlist[position])
            maxvalue = (np.max(self.FRD2dresiduals[positionindex]))
            absminvalue = np.max(np.multiply(-1,(self.FRD2dresiduals[positionindex])))
            absmax = max(maxvalue,absminvalue)
            plt.imshow(self.FRD2dresiduals[positionindex],vmin=-absmax,vmax=absmax,origin='lower')
            plt.colorbar()
            plt.title('Position: {}'.format(positionindex))
            plt.show()