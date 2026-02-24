"""Class to load all data saved at timeStep of an openFoam simulation
=====================================================================

.. autoclass:: OpenFoamSimu

.. automethod:: OpenFoamSimu.keys

.. automethod:: OpenFoamSimu.readopenfoam

"""

import os, sys
import subprocess
import numpy as np
from fluidfoam import readmesh, readfield, OpenFoamFile, readscalar , readvector , readsymmtensor
import netCDF4 as nc

class Error(Exception):
    pass

class DirectorySimuError(Error):
    def __init__(self, simu):
        super(DirectorySimuError,self).__init__(
                "No directory found for simulation named {}".format(simu))

class OpenFoamSimu(object):
    """
    Class to load all data saved at timeStep of an openFoam simulation

    Args:
        path: str, reference path where simulations are stored.\n
            You may want to provide path if all your simulations are located
            inside path and subfolders of path. You can do it by modifying
            in the __init__ path='/path/to/the/simulations/'\n
        simu: str, name of the simu that has to be loaded.\n
            If simu=None, it will lists all existing simulation names in path
            and ask you to choose.\n
        timeStep: str, timeStep to load. If None, load the last time step\n
        structured: bool, true if the mesh is structured\n
        dataToLoad: list of str, list containing the name of the varaibles 
            to read and load. If None, read and load all saved variables.
    """

    def __getattr__(self, name):
        """
        The user can write simu.U0 instead of simu.U[0]
        """
        # ex: U0, U1 ... 
        if name[-1].isdigit():
            field_name = name[:-1]
            idx = int(name[-1])

            if field_name in self.__dict__:
                field = self.__dict__[field_name]
                return field[idx]

        raise AttributeError(
            f"{self.__class__.__name__} has no attribute '{name}'"
        )

    def add_variable(self, name, value):

        python_name = name.replace('.', '_').replace('Mean','bar').replace(':','_')
        #Python attribute
        setattr(self, python_name, value)

        #OpenFoam attribute 
        if python_name != name:
            setattr(self, name, value)

        #Add attribute only if not in self.variables
        if python_name not in self.variables:
            self.variables.append(python_name)



    def __init__(self,  path=None, boundary = None, simu=None, timeStep=None, structured=False,
                dataToLoad=None,NetCDF = True, precision=15, order='F'):
        
        if boundary is None : 
            boundary = ['cylinder_1' , 'cylinder_2','bottom']


        if path == None and simu == None:
            # If nothing if given, consider the current directory as the 
            # simulation to load
            self.directory = os.getcwd()+'/'
            self.simu = os.getcwd().split('/')[-1]
            path = './'

        elif simu == None:
            # If only path is provided, consider all subfolders as possible
            # simulations to load
            self.directory = self._choose_simulation(path)
            self.simu = self.directory.split("/")[-2]
            path = './'

        else:
            # If path and simu are provided, consider the given directory
            # as the simulation to load
            self.simu = simu
            if path.endswith('/') is False:
                path += '/'
            self.directory = path + simu

            if self.directory.endswith('/') is False: 
                self.directory += '/'
    
        if NetCDF is True : 
            """ If one wants to write and read fileds in netCDF file """

            filename = "fields.nc"
            if not os.path.exists(self.directory + '/' + filename) :
                self.readmesh(structured=structured, precision=precision,boundary = boundary,
                order=order)
                self.readopenfoam(timeStep=timeStep,boundary = boundary, structured=structured, 
                    dataToLoad=dataToLoad, precision=precision,
                    order=order)
        
                self.writeNetCDF(self.directory,boundary)
                self.readNetCDF(self.directory)
            
            else :
                self.readNetCDF(self.directory)
        
        
        else : 
            """ If one wants to read fields directly without reading a netCDF file """

            self.readmesh(boundary = boundary,timeStep=timeStep, structured=structured, 
                    precision=precision, order=order)

            self.readopenfoam(timeStep=timeStep,boundary = boundary, structured=structured, 
                        dataToLoad=dataToLoad, precision=precision,
                        order=order)



    def writeNetCDF(self, path, boundary):

        filename = "fields.nc"
        ncfile = nc.Dataset(os.path.join(path, filename), "w", format="NETCDF4")

        # --- Mesh coordinates --- #
        ncfile.createDimension("cells", len(self.x))
        for name, data in zip(["x","y","z"], [self.x, self.y, self.z]):
            var = ncfile.createVariable(name, "f4", ("cells",))
            var[:] = data

        # --- Boundaries coordinates --- #
        for bound in boundary:
            xc, yc, zc = readmesh(path, boundary=bound)
            dim_name = f"{bound}_dim"
            ncfile.createDimension(dim_name, len(xc))
            for name, data in zip([f"x_{bound}", f"y_{bound}", f"z_{bound}"], [xc, yc, zc]):
                var = ncfile.createVariable(name, "f4", (dim_name,))
                var[:] = data

        # --- Global dimensions for vectors/tensors --- #
        if "nComp" not in ncfile.dimensions: ncfile.createDimension("nComp", 3)
        if "nSymm" not in ncfile.dimensions: ncfile.createDimension("nSymm", 6)
        if "nTensor" not in ncfile.dimensions: ncfile.createDimension("nTensor", 9)

        # --- Save fields --- #
        for field in self.variables:

            # Skip field if the file does not exist
            file_path = os.path.join(self.directory, self.timeStep, field)
            if not os.path.exists(file_path):
                continue

            data = getattr(self, field)
            ndim = np.shape(data)[0]

            # --- VolScalarField --- #
            if ndim == len(self.x):
                var = ncfile.createVariable(field, "f4", ("cells",))
                var[:] = data

            # --- SurfaceScalarField --- #
            elif ndim == 1:
                for bound in boundary:
                    s_data = readscalar(path, os.path.join(self.timeStep, field), boundary=bound)
                    dim_name = f"{field}_{bound}_dim"
                    if dim_name not in ncfile.dimensions:
                        ncfile.createDimension(dim_name, len(s_data))
                    var = ncfile.createVariable(f"{field}_{bound}", "f4", (dim_name,))
                    var[:] = s_data

            # --- VolVectorField --- #
            elif ndim == 3:
                # SurfaceVectorField
                if np.shape(data)[-1] == 1:
                    for bound in boundary:
                        s_data = readvector(path, os.path.join(self.timeStep, field), boundary=bound)
                        for i in range(3):
                            dim_name = f"{field}_{bound}{i}_dim"
                            if dim_name not in ncfile.dimensions:
                                ncfile.createDimension(dim_name, s_data.shape[1])
                            var_i = ncfile.createVariable(f"{field}_{bound}{i}", "f4", (dim_name,))
                            var_i[:] = s_data[i]
                # VolVectorField
                else:
                    for i in range(3):
                        dim_name = f"{field}{i}_dim"
                        if dim_name not in ncfile.dimensions:
                            ncfile.createDimension(dim_name, len(self.x))
                        var_i = ncfile.createVariable(f"{field}{i}", "f4", (dim_name,))
                        var_i[:] = data[i]

            # --- Symmetric Tensor --- #
            elif ndim == 6:
                for i in range(6):
                    dim_name = f"{field}{i}_dim"
                    if dim_name not in ncfile.dimensions:
                        ncfile.createDimension(dim_name, len(self.x))
                    var_i = ncfile.createVariable(f"{field}{i}", "f4", (dim_name,))
                    var_i[:] = data[i]

            # --- Tensor --- #
            elif ndim == 9:
                for i in range(9):
                    dim_name = f"{field}{i}_dim"
                    if dim_name not in ncfile.dimensions:
                        ncfile.createDimension(dim_name, len(self.x))
                    var_i = ncfile.createVariable(f"{field}{i}", "f4", (dim_name,))
                    var_i[:] = data[i]

        ncfile.close()
        print("Done writing NetCDF")


    def readNetCDF(self, path): 
        filename = "fields.nc"
        ncfile = nc.Dataset(path + filename , "r" , format="NETCDF4")

        # Récupère tous les noms de variables et les change
        self.variables = [v.replace('.', '_').replace('Mean','bar').replace(':','_') 
                        for v in ncfile.variables.keys()]

        # Lecture des données dans les attributs
        for python_var, orig_var in zip(self.variables, ncfile.variables.keys()):
            data = ncfile.variables[orig_var][:]
            setattr(self, python_var, data)

        ncfile.close()

	
	
    def readmesh(self,boundary = None ,timeStep=None, structured=False, precision=10, order='F'):
        
        if timeStep is None:
            dir_list = os.listdir(self.directory)
            time_list = []

            for directory in dir_list:
                try:
                    float(directory)
                    time_list.append(directory)
                except:
                    pass
            time_list.sort(key=float)
            timeStep = time_list[-1]

        elif type(timeStep) is int:
            #timeStep should be in a str format
            timeStep = str(timeStep)

        self.timeStep = timeStep

        # Check if cell center position is written in the output directory
        try:
            field = OpenFoamFile(path=self.directory, time_name=self.timeStep,
                                 name='C', structured=False, precision=precision,
                                 order=order)
            values = field.values
            shape = (3, values.size // 3)
            values = np.reshape(values, shape, order=order)
            if structured and not field.uniform:
                try:
                    values[0:3, :] = values[0:3, self.ind]
                    shape = (3,) + tuple(self.shape)
                    values = np.reshape(values, shape, order=order)
                except:
                    print("Variable {} could not be loaded".format(var))
                    self.variables.remove(var)
            X, Y, Z = values[0], values[1], values[2]

            # # --- Boundaries coordinates --- #
            for bound in boundary :
                field = OpenFoamFile(path=self.directory, 
                                time_name=self.timeStep,
                                name='C', 
                                boundary = bound,
                                structured=False, 
                                precision=precision,
                                order=order)
                values = field.values
                shape = (3, values.size // 3)
                values = np.reshape(values, shape, order=order)
        
                xc, yc, zc = values[0],values[1],values[2]
                setattr(self,f'x_{bound}',xc)
                setattr(self,f'y_{bound}',yc)
                setattr(self,f'z_{bound}',zc)

        except FileNotFoundError:
            X, Y, Z = readmesh(self.directory, boundary = None, structured=structured,
                            precision=precision, order=order)
        self.x = X
        self.y = Y
        self.z = Z
        if structured:
            nx = np.unique(X).size
            ny = np.unique(Y).size
            nz = np.unique(Z).size
            self.ind = np.array(range(nx*ny*nz))
            self.shape = (nx, ny, nz)

        # # --- Boundaries coordinates --- #
        for bound in boundary :
            xc,yc,zc = readmesh(self.directory, boundary = bound, structured=structured,
                            precision=precision, order=order)
            
            setattr(self,f'x_{bound}',xc)
            setattr(self,f'y_{bound}',yc)
            setattr(self,f'z_{bound}',zc)

        





    def readopenfoam(self, boundary, timeStep=None, structured=False, dataToLoad=None,
                     precision=10, order='F'):
        """
        Reading SedFoam results
        Load the last time step saved of the simulation

        Args:
            timeStep : str or int, timeStep to load. If None, load the last time step\n
            structured : bool, true if the mesh is structured
        """

        if timeStep is None:
            dir_list = os.listdir(self.directory)
            time_list = []

            for directory in dir_list:
                try:
                    float(directory)
                    time_list.append(directory)
                except:
                    pass
            time_list.sort(key=float)
            timeStep = time_list[-1]

        elif type(timeStep) is int:
            #timeStep should be in a str format
            timeStep = str(timeStep)

        self.timeStep = timeStep

        #List all variables saved at the required time step removing potential
        #directory that cannot be loaded
        if dataToLoad is None:
            self.variables = []
            basepath = self.directory+self.timeStep+'/'
            for fname in os.listdir(basepath):
                path = os.path.join(basepath, fname)
                if os.path.isdir(path):
                    # skip directories
                    continue
                else:
                    self.variables.append(fname)
                    
            #Remove C, Cx, Cy and Cz if present
            var_to_remove = ['C', 'Cx', 'Cy', 'Cz']
            for var in var_to_remove:
                if var in self.variables:
                    self.variables.remove(var)
        else:
            self.variables = dataToLoad
        

        for var in self.variables:
            #Check if file is in path 
            file_path = os.path.join(self.directory, self.timeStep, var)
            if not os.path.exists(file_path):
                continue           

            field = OpenFoamFile(
                path=self.directory,
                time_name=self.timeStep,
                name=var,
                structured=False,
                precision=precision,
                order=order
            )
            values = field.values

            # ---- volume scalar ---- #
            if field.type_data == "scalar":
                if structured and not field.uniform:
                    try:
                        values = np.reshape(values[self.ind], self.shape, order=order)
                    except:
                        print(f"Variable {var} could not be loaded")
                        self.variables.remove(var)
                        continue
                self.add_variable(var, values)

                # ---- surface scalar ---- #
                if np.shape( getattr(self, var) )[-1] == 1 : 
                    for bound in boundary:
                        try:
                            s_values = readscalar(
                                path=self.directory,
                                time_name=self.timeStep,
                                name=var,   
                                boundary=bound
                            )
                            self.add_variable(f"{var}_{bound}", s_values)
                        except FileNotFoundError:
                            continue

            # ---- vector fields ---- #
            elif field.type_data == "vector":
                # volume vector
                shape = (3, values.size // 3)
                values = np.reshape(values, shape, order=order)
                if structured and not field.uniform:
                    try:
                        values[0:3, :] = values[0:3, self.ind]
                        shape = (3,) + tuple(self.shape)
                        values = np.reshape(values, shape, order=order)
                    except:
                        print(f"Variable {var} could not be loaded")
                        self.variables.remove(var)
                        continue
                
                # Add field
                self.add_variable(var, values)

                # Add all components
                for i in range(values.shape[0]):
                    self.add_variable(f"{var}{i}", values[i])
                

                # surface vector
                if np.shape( getattr(self, var) )[-1] == 1 :
                    for bound in boundary:
                        try:
                            s_values = readvector(
                                path=self.directory,
                                time_name=self.timeStep,
                                name=var,        
                                boundary=bound
                            )
                            
                            #Components
                            for i in range(s_values.shape[0]):
                                self.add_variable(f"{var}_{bound}{i}", s_values[i])
                        except FileNotFoundError:
                            continue

            # ---- symmtensor fields ---- #
            elif field.type_data == "symmtensor":
                shape = (6, values.size // 6)
                values = np.reshape(values, shape, order=order)
                if structured and not field.uniform:
                    try:
                        values[0:6, :] = values[0:6, self.ind]
                        shape = (6,) + tuple(self.shape)
                        values = np.reshape(values, shape, order=order)
                    except:
                        print("Variable {} could not be loaded".format(var))
                        self.variables.remove(var)
                        continue
                        
                # Add field
                self.add_variable(var, values)

                # Add all components
                for i in range(values.shape[0]):
                    self.add_variable(f"{var}{i}", values[i])

            # ---- tensor fields ---- #
            elif field.type_data == "tensor":
                shape = (9, values.size // 9)
                values = np.reshape(values, shape, order=order)
                if structured and not field.uniform:
                    try:
                        values[0:9, :] = values[0:9, self.ind]
                        shape = (9,) + tuple(self.shape)
                        values = np.reshape(values, shape, order=order)
                    except:
                        print("Variable {} could not be loaded".format(var))
                        self.variables.remove(var)
                        continue
                        
                # Add field
                self.add_variable(var, values)

                # Add all components
                for i in range(values.shape[0]):
                    self.add_variable(f"{var}{i}", values[i])

            #Set attributes
            self.__setattr__(var.replace('.', '_').replace('Mean','bar').replace(':','_'), values)

    def keys(self):
        """
        Print the name of all variables loaded from simulation results
        """
        print("Loaded available variables are :")
        print(self.variables)

    def _choose_simulation(self, path):
        """
        Make a list of all directories located in path containing a simulation.
        Ask the user which simulation to load

        Args:
            path : str, reference path where simulations are stored.
        """
        directories = []
        subDirectories = [x[0] for x in os.walk(path)]

        for f in subDirectories:
            #A directory is detected to be a simulation if it contains a 0_org/ folder
            if f + "/constant" in subDirectories:
                directories.append(f)

        # If no directories found
        if len(directories) == 0:
            raise DirectorySimuError(path)

        for i in range(len(directories)):
            print("{} : {}".format(i, directories[i]))
        chosenSimulation = -1
        while type(chosenSimulation) is not int or (
                chosenSimulation < 0 or chosenSimulation > len(directories) - 1):
            chosenSimulation = int( input(
                "Please, choose one simulation ! (integer between {} and {})".format(
                    0, len(directories) - 1))
            )
        directory = directories[chosenSimulation]

        return directory + "/"

    def _find_directory(self, path, simu):
        """
        Look for the directory of simu in all the sub directories of path. If several
        directories are found, the program asks which directory is the good one.

        Args:
            path : str, reference path where simulations are stored.
            simu : str, name of the simu that has to be loaded. If None, it will
                lists all existing simulation names in path and ask you to choose
        """
        directories = []
        subDirectories = [x[0] for x in os.walk(path)]

        for f in subDirectories:
            if f.endswith(simu):
                directories.append(f)

        # If no directories found
        if len(directories) == 0:
            raise DirectorySimuError(simu)

        # If several directories found, ask for the one wanted
        elif len(directories) > 1:
            print("The following simulations has been found :")
            for i in range(len(directories)):
                print("{} : {}".format(i, directories[i]))
            chosenSimulation = -1
            while type(chosenSimulation) is not int or (
                    chosenSimulation < 0 or chosenSimulation > len(directories) - 1):
                chosenSimulation = int(input(
                    "Please, choose one simulation ! (integer between {} and {})".format(
                        0, len(directories) - 1)
                    )
                )
            directory = directories[chosenSimulation]

        else:
            directory = directories[0]

        return directory + "/"

if __name__ == "__main__":

    simu = "box"
    timeStep = "4"

    for d in dirs:
        rep = os.path.join(os.path.dirname(__file__), "../output_samples")

        mySimu = OpenFoamSimu(path=rep, simu=simu, timeStep=timeStep, structured=True)

        mySimu.keys()

        mySimu.U

