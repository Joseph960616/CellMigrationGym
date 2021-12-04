from xml.dom import minidom 
import os  
import numpy as np

#Pre-define parameters
EMPTY_CELL = 20
AGENT_CELL = 1
class Create_XML():
    def __init__(self):
        #create file 
        self.root = minidom.Document() 
        self.xml = self.root.createElement('mujoco')  
        self.root.appendChild(self.xml) 
        #set world configuration  
        self.timestep = self.root.createElement('option') 
        self.timestep.setAttribute('timestep', '1')
        self.xml.appendChild(self.timestep)
        self.timestep = self.root.createElement('size') 
        self.timestep.setAttribute('njmax', '8000')
        self.timestep.setAttribute('nconmax', '8000')
        self.xml.appendChild(self.timestep)

        # #load data of the cells
        # self.n = len(open(file_path).readlines())
        # count = 0
        # location = [0] * self.n
        # radius = [0] * self.n
        # with open(file_path) as file:
        #     for line in file:
        #         line = line[:len(line)-1]
        #         vec = line.split(', ')
        #         id = int(vec[0])
        #         location[count] = np.array(((vec[5]), (vec[6]), (vec[7])))
        #         radius[count] = float(vec[8])/2
        #         cell_name = vec[9]
        #         count += 1
        # self.pos = location
        # self.radius = radius

    def create_world(self):
        #create world
        worldbody = self.root.createElement('worldbody') 
        self.xml.appendChild(worldbody)
        actuator = self.root.createElement('actuator') 
        self.xml.appendChild(actuator)
        # #create floor
        # body = self.root.createElement('body') 
        # body.setAttribute('name', 'floor')
        # body.setAttribute('pos', '250 250 -10')
        # worldbody.appendChild(body)
        # floor = self.root.createElement('geom') 
        # floor.setAttribute('condim', '3')
        # floor.setAttribute('rgba', '0 1 0 1')
        # floor.setAttribute('size', '500 500 0.02')
        # floor.setAttribute('type', 'box')
        # body.appendChild(floor)

        #create pre-define empty cell
        body = [0] * EMPTY_CELL
        for i in range(EMPTY_CELL):
            body[i] = self.root.createElement('body') 
            body[i].setAttribute('name', 'cell%01d'%i)
            body[i].setAttribute('pos', '0 0 0')
            worldbody.appendChild(body[i])
            
            #geom parameter
            geom = self.root.createElement('geom') 
            geom.setAttribute('mass', '1.0')
            geom.setAttribute('name', 'geo_cell%01d'%i)
            geom.setAttribute('rgba', '0 0 0 0')
            geom.setAttribute('size', '5')
            geom.setAttribute('type', 'sphere')
            body[i].appendChild(geom)
        

    def export_model(self):   
        xml_str = self.root.toprettyxml(indent ="\t")  
        save_path_file = "./Embryo/utils/cell_neighbour.xml"
        with open(save_path_file, "w") as f: 
            f.write(xml_str)
        return xml_str
    
    def run(self):
        self.create_world()
        model = self.export_model()
        return model

if __name__ == '__main__':
    m = Create_XML()
    m.run()