#Import all the libraries
import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup #Beautiful Soup library for parsing data in python 

###############################ASAM OPEN DRIVE PARSING FUNCTION#########################################
##Function to extract all the data from the open drive file#############################################
def file_parser(file_name):
    with open(file_name,'r') as f: #now we are going to read each line inside the file
        data = f.read()  #Read all the file
    soup = BeautifulSoup(data,'xml') #store the data into the soup
    return soup
#########################################################################################################

###Data Visualization of road geometry inside the OpenDrive ASAM file
def center_visualizer(file_name, graph_num): #File parsing function 
    soup = file_parser(file_name)
    road = soup.find_all('road')[:-1] #first find all the road segments and ignore the last one
    temp = []
    for elem in road: #traverse each road segment in the open drive file
        planview = elem.find('planView') #we are going to find the planview
        road_geom = planview.find_all('geometry') #then find all the geometry
        x = [] #list of x-coordinates
        y = [] #list of y-coordinates
        
        for rg in road_geom: #traverse each geometry for each road
            temp_x = float(rg['x']) #don't forget to cast this to float!
            temp_y = float(rg['y']) #don't forget to cast this float!
            x.append(temp_x) #add the x coordinates to the x list
            y.append(temp_y) #add the y coordinates to the y list
            hdg = float(rg['hdg']) #gets the heading of the road
            length = float(rg['length']) #gets the length of the road
           
            if (rg.line): #we want to check if this is the line segment
                xi = temp_x #x-coordinate starting point
                xf = temp_x + np.cos(hdg) * length #x-coordinate ending point
                yi = temp_y #y-coordinate starting point
                yf = temp_y + np.sin(hdg) * length #y-coordinate ending point
                x_line = [xi,xf] #x-coordinates on the line segment
                y_line = [yi,yf] #y-coordinates on the line segment
                plt.plot(x_line,y_line,color='blue') #let's plot the straight line segment
                temp.append('line')
                
            else: #if this is not a line segment then this is a curve
                #IMPORTANT #NOTE_TO KEEP IN MIND OF: 
                #if curvature is negative:
                #then the center is at the right hand side of the vehicle in vehicle's frame

                curvature = float(rg.find('arc')['curvature']) #curvature
                radius = np.abs(1/curvature) #radius = 1/curvature
                arc_length = float(rg['length']) #arc-length
                
                ############WE ARE GOING TO DETERMINE THE VECTOR DIRECTION####################
                T = np.array([np.cos(hdg) , np.sin(hdg),0]) #tangent vector on the curved line
                N = np.array([0,0,1]) #normal vector on the curved line
                S = np.cross(N,T) #cross product of normal and tangent vectors

                xi = temp_x #initial x-coordinate
                yi = temp_y #initial y-coordinate 
                center = [] #center coordinates

                if curvature < 0: #if the curvature is negative then it it CLOCKWISE
                    center = np.array([xi,yi]) - S[0:2]*radius #center is equal to the current point subtract the radius
                else: #if the curvature is positive then it is COUNTER-CLOCKWISE
                    center = np.array([xi,yi]) + S[0:2]*radius #center is equal to the current point add the radius
                
                x_diff = xi - center[0] #x_diff means the difference between the current x-coordinate and the center x-coordinate
                y_diff = yi - center[1] #y_diff means the difference between the current y-coordinate and the center y-coordinate
                
                initial_theta = np.arctan2(y_diff, x_diff) #initial theta is the angle between the y_diff and x_diff
                delta_theta = abs(arc_length/radius) #delta theta is the final angle of the curvature

                if curvature < 0: #if the curvature is negative
                    final_theta = initial_theta - delta_theta  #we want to go the opposite way so subtract delta theta from initial theta
                else: #this condition means that the curvvature is positive
                    final_theta = initial_theta + delta_theta  #we want to go COUNTER-CLOCKWISE so ADD the delta theta to initial theta
                
                temp_theta_range = np.sort(np.array([initial_theta,final_theta])) #range between intial theta and final theta
                theta_range = np.linspace(temp_theta_range[0],temp_theta_range[len(temp_theta_range)-1],10) #we are going to split the plots in to multiple sections to make the curve look more smooth
                x_curve = [] #x-coordinates on the curve
                y_curve = [] #y-coordinates on the curve
                
                for i in (theta_range): #for the theta in range
                    x_append = center[0] + radius*np.cos(i) #set the x-coordinate of the curve to be the center + cos of the radius
                    y_append = center[1] + radius*np.sin(i) #set the y-coordinate of the curve to be the center + sin of the radius
                    x_curve.append(x_append) #add this into the list of x-curve points
                    y_curve.append(y_append) #add this into the list of y-curve points
                plt.plot(x_curve,y_curve,color='blue') #plot the x and y coordinates for the curve
                temp.append('curve')

        assert(len(x) == len(y)) #just to make sure the size of x and y list are the same
        scatter = plt.scatter(x,y,color='red') #scatter the plots
        plt.xlabel('x position in meters') #x-coordinates label
        plt.ylabel('y position in meters') #y-coordinates label
        plt.title('Race Track Layout ' + str(graph_num)) #title of the coordinate
    
    return scatter #return the graph plot of the tracks in the ASAM OpenDrive File