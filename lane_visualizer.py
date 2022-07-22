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

######################Straight Line plotting helper function#########################
def line_plot_utility(xi,xf,yi,yf,style,p_color,alpha,N):
    new_vect_i = np.array([xi,yi]) + alpha*N
    new_vect_f = np.array([xf,yf]) + alpha*N
    x_scatter = [new_vect_i[0],new_vect_f[0]]
    y_scatter = [new_vect_i[1],new_vect_f[1]]
    plt.plot(x_scatter,y_scatter,style,color=p_color,linewidth = 1.0)

######################Curve plotting helper function#########################
def curve_plot_utility(theta_range,center,temp_radius,style,p_color):
    x_curve = []
    y_curve = []
    for i in theta_range: 
        x_append = center[0] + temp_radius*np.cos(i)
        y_append = center[1] + temp_radius*np.sin(i)
        x_curve.append(x_append) #add this into the list of x-curve points
        y_curve.append(y_append) #add this into the list of y-curve points
    plt.plot(x_curve, y_curve,style,color=p_color,linewidth=1.0)


#########################Check if we can get all the lane data from this tag from the ASAM Open DRIVE File
def check_valid_side(side):
    lane_array = []
    temp_lanes = []
    alpha_array = []
    if side is not None:
        temp_lanes = side.find_all('lane') #let's find all the left lane
        for tl in temp_lanes: #traverse each element in the left lane
            lane_array.append(tl['type']) #add each of the left segment to the temporary left array
            temp_alpha = tl.find('width')
            alpha_array.append(float(temp_alpha['a']))
        return True,lane_array,alpha_array
    return False,[],[]

    #############################TRAVERSE EACH ELEMENT IN THE ROAD################################
def plotter(road,graph_num):
    for elem in road: #traverse each road segment in the open drive file
        planview = elem.find('planView') #we are going to find the planview
        road_geom = planview.find_all('geometry') #then find all the geometry
        lane_types = elem.find('lanes') #let's find all the lanes

        ###################LEFT SIDE OF THE ROAD###################################################
        left = lane_types.find('left') #find all the left side of the road
        left_valid, left_array,left_alpha_array = check_valid_side(left) #check the left side of the lane

        #################RIGHT SIDE OF THE ROAD#####################################################
        right = lane_types.find('right') #find all the center side of the road
        right_valid, right_array,right_alpha_array = check_valid_side(right) #check the right side of the lane
        ###########################################################################################
        
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
                x_line = [xi,xf]
                y_line = [yi,yf]

                #####################NORMAL VECTOR#######################
                delta_x = xf - xi #find the difference between two x-coordinates
                delta_y = yf - yi #find the differences between two y-coordinates
                
                N_left = [-delta_y,delta_x]/(np.sqrt(delta_x**2 + delta_y**2)) #Normal Vector for left side
                N_right = [delta_y,-delta_x]/(np.sqrt(delta_x**2 + delta_y**2)) #Normal Vector for right side

                ########################LEFT ARRAY#######################
                left_alpha_val = 0
                left_idx = 0
                assert (len(left_array) == len(left_alpha_array))
                if left_valid: 
                    for left_elem in left_array:  
                        if left_elem == 'sidewalk':
                            color = 'black'
                            style = '-'
                            left_alpha_val += left_alpha_array[left_idx]
                            line_plot_utility(xi,xf,yi,yf,style,color,left_alpha_val,N_left)

                        elif left_elem == 'driving':
                            color = 'blue'
                            style = '--'
                            left_alpha_val += 0.5*left_alpha_array[left_idx]
                            line_plot_utility(xi,xf,yi,yf,style,color,left_alpha_val,N_left)
                            left_alpha_val -= 0.5*left_alpha_array[left_idx]
                            left_alpha_val += left_alpha_array[left_idx]
                            
                        left_idx +=1 #increment left index
                
                
                ###################RIGHT ARRAY#########################
                right_alpha_val = 0
                right_idx = 0
                assert (len(right_array) == len(right_alpha_array))
                if right_valid: 
                    for right_elem in right_array: 
                        if right_elem == 'sidewalk':
                            color = 'black'
                            style = '-'
                            right_alpha_val += right_alpha_array[right_idx]
                            line_plot_utility(xi,xf,yi,yf,style,color,right_alpha_val,N_right)
                        elif right_elem == 'driving':
                            color = 'blue'
                            style = '--'
                            right_alpha_val += 0.5*right_alpha_array[right_idx]
                            line_plot_utility(xi,xf,yi,yf,style,color,right_alpha_val,N_right)
                            right_alpha_val -= 0.5*right_alpha_array[right_idx]
                            right_alpha_val += right_alpha_array[right_idx]
                            
                        right_idx +=1

                ######PLOT THE POINTS ALONG THE LINE SEGMENTS#############
                plt.plot(x_line,y_line,color='red')
                ##########################################################
                
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
                curvature_direction = [] #store the direction of each curvature

                if curvature < 0: #if the curvature is negative then it it CLOCKWISE
                    center = np.array([xi,yi]) - S[0:2]*radius #center is equal to the current point subtract the radius
                    curvature_direction.append(False)
                else: #if the curvature is positive then it is COUNTER-CLOCKWISE
                    center = np.array([xi,yi]) + S[0:2]*radius #center is equal to the current point add the radius
                    curvature_direction.append(True)
                
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
                
                for i in theta_range: #for the theta in range
                    x_append = center[0] + radius*np.cos(i) #set the x-coordinate of the curve to be the center + cos of the radius
                    y_append = center[1] + radius*np.sin(i) #set the y-coordinate of the curve to be the center + sin of the radius
                    x_curve.append(x_append) #add this into the list of x-curve points
                    y_curve.append(y_append) #add this into the list of y-curve points
                plt.plot(x_curve,y_curve,color='red') #plot the x and y coordinates for the curve

                #####THIS IS WHERE I AM GOING TO ADJUST EACH LANE SEGMENT OF THE CURVE

                #########################LEFT ARRAY###################################
                left_idx = 0
                assert (len(left_array) == len(left_alpha_array))
                temp_radius = radius
                if left_valid:
                    for left_elem in left_array:
                        for curve_dir in curvature_direction:
                            if curve_dir == True:
                                if left_elem == 'driving':
                                    color = 'blue'
                                    style = '--'
                                    temp_radius -= 0.5*left_alpha_array[left_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                                    temp_radius += 0.5*left_alpha_array[left_idx]
                                    temp_radius -= left_alpha_array[left_idx]

                                elif left_elem == 'sidewalk':
                                    color = 'black'
                                    style = '-'
                                    temp_radius -= left_alpha_array[left_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                                    
                            else:
                                if left_elem == 'driving':
                                    color = 'blue'
                                    style = '--'
                                    temp_radius += 0.5*left_alpha_array[left_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                                    temp_radius -= 0.5*left_alpha_array[left_idx]
                                    temp_radius += left_alpha_array[left_idx]
                                    
                                elif left_elem == 'sidewalk':
                                    color = 'black'
                                    style = '-'
                                    temp_radius += left_alpha_array[left_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                         
                            left_idx +=1  #increment the left index
                
                # ###################RIGHT ARRAY#########################
                right_idx = 0
                assert (len(right_array) == len(right_alpha_array))
                temp_radius = radius
                if right_valid:
                    for right_elem in right_array:
                        for curve_dir in curvature_direction:
                            if curve_dir == True:
                                if right_elem == 'driving':
                                    color = 'blue'
                                    style = '--'
                                    temp_radius += 0.5*right_alpha_array[right_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                                    temp_radius -= 0.5*right_alpha_array[right_idx]
                                    temp_radius += right_alpha_array[right_idx]

                                elif right_elem == 'sidewalk':
                                    color = 'black'
                                    style = '-'
                                    temp_radius += right_alpha_array[right_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                                    
                            else:
                                if right_elem == 'driving':
                                    color = 'blue'
                                    style = '--'
                                    temp_radius -= 0.5*right_alpha_array[right_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                                    temp_radius += 0.5*right_alpha_array[right_idx]
                                    temp_radius -= right_alpha_array[right_idx]
                                    
                                elif right_elem == 'sidewalk':
                                    color = 'black'
                                    style = '-'
                                    temp_radius -= right_alpha_array[right_idx]
                                    curve_plot_utility(theta_range,center,temp_radius,style,color)
                         
                            right_idx +=1  #increment the right index

                       
        assert(len(x) == len(y)) #just to make sure the size of x and y list are the same
        scatter = plt.scatter(x,y,color='red') #scatter the plots
        plt.xlabel('x position in meters') #x-coordinates label
        plt.ylabel('y position in meters') #y-coordinates label
        plt.title('Race Track Layout '+str(graph_num)) #title of the coordinate
    return scatter

    #Function to generate lane data visualization while parsing the ASAM Open DRIVE file
def lane_visualizer(file_name,graph_num): 
    ##############################FILE PARSING PROCESS############################################
    soup = file_parser(file_name)
    road = soup.find_all('road')[:-1] #first find all the road segments and ignore the last one
    ##############################################################################################
    scatter = plotter(road,graph_num)
    return scatter #return the graph plot of the tracks in the ASAM OpenDrive File