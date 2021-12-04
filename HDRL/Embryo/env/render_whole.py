
    def _render(self, mode='human', img_resolution = 128):
        agent = self.agent
        cell = self.cell
        orientation = [0,0,0,1]
        img_start = img_resolution // 4 + 1             # cutting between [33,97)
        img_end = img_resolution* 3 // 4 + 1
        #Disable all rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.changeVisualShape(agent[0],-1,rgbaColor=[1, 0, 0, 1])     #AI cell -> Red
        p.changeVisualShape(agent[1],-1,rgbaColor=[0, 1, 0, 1])     #Target cell -> Green
        # for ce in cell:
        #     p.changeVisualShape(ce,-1,rgbaColor=[0, 0, 1, 1])       #State cell -> Blue

        p.resetDebugVisualizerCamera(cameraDistance=80, \
                cameraYaw=58, \
                cameraPitch=-90.01, \
                cameraTargetPosition=self.pos_a[0][self.ai_first_appear_index])
        
        for i in range(len(self.pos_interpolations_neighbour_a)):       #one stage
            for k in range(5):                                          #one timestep
                r_ai_cell = self.radius_target_a[i][0]
                is_neighbour = np.zeros(len(self.cell_name_interpolations_neighbour_a[0][:,k]))
                for m in range(len(self.cell_name_interpolations_neighbour_a[0][:,k])):
                    dist = np.linalg.norm(self.pos_interpolations_target_a[i][0][k]-self.pos_interpolations_neighbour_a[i][m][k])
                    r_cell = self.radius_neighbour_a[0][m]
                    is_neighbour[m] = self.neighbor_model.predict([[dist, r_ai_cell, r_cell, len(self.pos_interpolations_a[i][:,k])]])
                #Disable rendering during creation.
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                #AI cell
                # Simulation data
                # p.resetBaseVelocity(agent[0],self.actions[i * self.tick_resolution + k])
                # p.resetBaseVelocity(agent[0],[np.random.randint(3),np.random.randint(3),np.random.randint(3)])
                # Observation data
                p.resetBasePositionAndOrientation(agent[0],self.pos_interpolations_target_a[i][0][k],orientation)
                p.resetDebugVisualizerCamera(cameraDistance=80, \
                                                cameraYaw=58, \
                                                cameraPitch=-90.01, \
                                                cameraTargetPosition=self.pos_interpolations_target_a[i][0][k])
                #Target cell
                p.resetBasePositionAndOrientation(agent[1],self.pos_interpolations_target_a[i][1][k],orientation)
                for j in range(len(self.cell_name_interpolations_neighbour_a[i])):
                    #Observational state cell
                    if self.cell_name_interpolations_neighbour_a[i][j][k] in self.state_cell_list:
                        p.resetBasePositionAndOrientation(cell[j],self.pos_interpolations_neighbour_a[i][j][k],orientation) 
                        if is_neighbour[j] == 1:
                            p.changeVisualShape(cell[j],-1,rgbaColor=[0, 0, 0.5, 1])    #
                        else:
                            p.changeVisualShape(cell[j],-1,rgbaColor=[0, 0, 1, 1])      #Blue for state cell
                    #Subgoal cell
                    if self.cell_name_interpolations_neighbour_a[i][j][k] in NEIGHBOR_CANDIDATE_1 and self.subgoal_stage == 0:
                        p.changeVisualShape(cell[j],-1,rgbaColor=[1, 1, 1, 1])
                    elif self.cell_name_interpolations_neighbour_a[i][j][k] in NEIGHBOR_CANDIDATE_2 and self.subgoal_stage == 1:
                        p.changeVisualShape(cell[j],-1,rgbaColor=[1, 1, 1, 1])
                    elif self.cell_name_interpolations_neighbour_a[i][j][k] in NEIGHBOR_CANDIDATE_3 and self.subgoal_stage == 2:
                        p.changeVisualShape(cell[j],-1,rgbaColor=[1, 1, 1, 1])

                # #Enable rendering
                # _,_,_,_,_,_,_,_,yaw,pitch,dist,target = p.getDebugVisualizerCamera()
                # print("yaw = {},pitch = {},dist = {},target = {}".format(yaw,pitch,dist,target))
                p.stepSimulation()
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                img_arr = p.getCameraImage(img_resolution,img_resolution)
                rgb = img_arr[2][img_start:img_end, img_start:img_end]
                depth = img_arr[3][img_start:img_end, img_start:img_end]
                depth_map = depth_conversion(depth)
                gray = rgb2gray(rgb)
                plt.figure(1)
                plt.title("64*64 Grey scale image")
                image = plt.imshow(gray, cmap='Greys_r')
                plt.figure(2)
                plt.title("64*64 Depth image")
                image = plt.imshow(depth,cmap='Greys_r')   # twilight_r
                # print("stage%d"%(i*10+k)) 

        return [gray,depth_map]