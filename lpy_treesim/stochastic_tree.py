"""
Defines the abstract class BasicWood, class Wire and class Support.
"""

from abc import abstractmethod
from openalea.plantgl.all import *
import copy
import numpy as np
from openalea.plantgl.scenegraph.cspline import CSpline
import random as rd

import collections
eps = 1e-6

from abc import ABC, abstractmethod
# class Tree():
#   #branch_dict = {}
#   trunk_dict = {}
#   """ This class will have all the parameters required to grow the tree, i.e. the transition
#   prob, max trunk length, max branch length etc. Each tree will have its own children branch and trunk classes """
#   def __init__(self):
#     self.trunk_num_buds_segment = 5
#     self.branch_num_buds_segment = 5
#     self.trunk_bud_break_prob = 0.5
#     self.branch_bud_break_prob = 0.5
#     self.num_branches = 0
#     self.num_trunks = 0
    
    
# # BRANCH AND TRUNK SUBCLASS OF WOOD		

class BasicWood(ABC):

  @staticmethod
  def clone(obj):
    try:
        return copy.deepcopy(obj)
    except copy.Error:
        raise copy.Error(f'Not able to copy {obj}') from None

  def __init__(self, copy_from = None, max_buds_segment: int = 5, thickness: float = 0.1,\
               thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,\
               tie_axis: list = (0,1,1),  order: int = 0, color: int = 0, material = 0, name: str = None):#,\
               #bud_break_prob_func: "Function" = lambda x,y: rd.random()):
                 
    #Location variables
    if copy_from:
      self.__copy_constructor__(copy_from)
      return
    self.start = Vector3(0,0,0)
    self.end = Vector3(0,0,0)
    #Tying variables    
    self.last_tie_location = Vector3(0,0,0)
    self.has_tied = False
    self.guide_points = []
    self.current_tied = False
    self.guide_target = -1#Vector3(0,0,0)
    self.tie_axis = tie_axis
    self.tie_updated = True
    #Information Variables
    self.__length = 0    
    self.age = 0    
    self.cut = False
    self.order = order
    self.num_branches = 0
    self.branch_dict = collections.deque()
    self.color = color
    self.material = material
    #Growth Variables
    self.max_buds_segment = max_buds_segment
    self.thickness = thickness
    self.thickness_increment = thickness_increment
    self.growth_length = growth_length
    self.max_length = max_length
    
    
    
  def __copy_constructor__(self, copy_from):
    update_dict = copy.deepcopy(copy_from.__dict__)
    for k,v in update_dict.items():
      setattr(self, k, v)
    #self.__dict__.update(update_dict)
    
  @abstractmethod
  def is_bud_break(self) -> bool:
    """This method defines if a bud will break or not -> returns true for yes, false for not. Input can be any variables"""
    pass
    #Example
    # prob_break = self.bud_break_prob_func(num_buds, self.num_buds_segment)
    # #Write dummy probability function 
    # if prob_break > self.bud_break_prob:
    #   return True
    # return False
  
  @abstractmethod
  def grow(self) -> None:
    """This method can define any internal changes happening to the properties of the class, such as reduction in thickness increment etc."""
    pass
  @property
  def length(self):
    return self.__length
  
  @length.setter
  def length(self, length):
    self.__length = min(length, self.max_length)
    
  def grow_one(self):
    self.age+=1
    self.length+=self.growth_length
    self.grow()
  
  @abstractmethod
  def create_branch(self) -> "BasicWood Object":
    """Returns how a new order branch when bud break happens will look like if a bud break happens"""
    pass
    #new_object = BasicWood.clone(self.branch_object)
    #return new_object
    #return BasicWood(self.num_buds_segment/2, self.bud_break_prob, self.thickness/2, self.thickness_increment/2, self.growth_length/2,\
              # self.max_length/2, self.tie_axis, self.bud_break_max_length/2, self.order+1, self.bud_break_prob_func)
    
  def update_guide(self, guide_target):
    curve = []
    self.guide_target = guide_target
    if self.guide_target == -1:
      return
    if self.has_tied == False:
      curve, i_target = self.get_control_points(self.guide_target.point, self.start , self.end, self.tie_axis)
    else:
      curve, i_target= self.get_control_points(self.guide_target.point, self.last_tie_location , self.end, self.tie_axis)
    if i_target:
      self.guide_points.extend(curve)
      #self.last_tie_location = copy.deepcopy(Vector3(i_target)) #Replaced by updating location at StartEach
  
  def tie_lstring(self, lstring, index):
    spline = CSpline(self.guide_points) 
    if str(spline.curve()) == "nan":
      raise ValueError("CURVE IS NAN", self.guide_points)
    remove_count = 0
    if not self.has_tied:
      if lstring[index+1].name in ['&','/','SetGuide']:
        del(lstring[index+1])
        remove_count+=1
      self.has_tied = True
    if lstring[index+1].name in ['&','/','SetGuide']:
      del(lstring[index+1])
      remove_count+=1
    lstring.insertAt(index+1, 'SetGuide({}, {})'.format(spline.curve(stride_factor = 100), self.length))
    return lstring,remove_count
  
  def tie_update(self):
    self.last_tie_location = copy.deepcopy(self.end)
    self.tie_updated = True

  def deflection_at_x(self,d, x, L): 
    """d is the max deflection, x is the current location we need deflection on and L is the total length"""
    return (d/2)*(x**2)/(L**3)*(3*L - x)
  #return d*(1 - np.cos(*np.pi*x/(2*L))) #Axial loading

 
  def get_control_points(self, target, start, current, tie_axis):
    pts = []
    Lcurve = np.sqrt((start[0]-current[0])**2 + (current[1]-start[1])**2 + (current[2]-start[2])**2)   
    if Lcurve**2 - (target[0]-start[0])**2*tie_axis[0] - (target[1]-start[1])**2*tie_axis[1] - (target[2]-start[2])**2*tie_axis[2]  <=0:
      return pts,None

    curve_end = np.sqrt(Lcurve**2 - (target[0]-start[0])**2*tie_axis[0]-(target[1]-start[1])**2*tie_axis[1] - (target[2]-start[2])**2*tie_axis[2])
   
   
    i_target = [target[0], target[1], target[2]]
    for j,axis in enumerate(tie_axis):
      if axis == 0:
        i_target[j] = start[j]+target[j]/abs(target[j])*(curve_end)
        break
    dxyz = np.array(i_target) - np.array(current)
    dx = np.array(current) - np.array(start)
    for i in np.arange(0.1,1.1,0.1):
      x = i#/Lcurve#+1#/(10*(Lcurve))
      
      d = self.deflection_at_x(dxyz, x*Lcurve, Lcurve)
      pts.append(tuple((start[0]+x*dx[0]+d[0],start[1]+x*dx[1]+d[1],start[2]+x*dx[2]+d[2])))
    return pts, i_target
      
# class Branch(BasicWood):
#   def __init__(self, num_buds_segment: int = 5, bud_break_prob: float = 0.8, thickness: float = 0.1,\
#                thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,\
#                tie_axis: tuple = (0,1,1), bud_break_max_length: int = 5, order: int = 0, bud_break_prob_func: "Function" = lambda x,y: rd.random()):
#     super().__init__(num_buds_segment, bud_break_prob, thickness, thickness_increment, growth_length,\
#                max_length, tie_axis, bud_break_max_length, order, bud_break_prob_func)
    
    
# class Trunk(BasicWood):
#   """ Details of the trunk while growing a tree, length, thickness, where to attach them etc """
#   def __init__(self, num_buds_segment: int = 5, bud_break_prob: float = 0.8, thickness: float = 0.1,\
#                thickness_increment: float = 0.01, growth_length: float = 1., max_length: float = 7.,\
#                tie_axis: tuple = (0,1,1), bud_break_max_length: int = 5, order: int = 0, bud_break_prob_func: "Function" = lambda x,y: rd.random()):
#      super().__init__(num_buds_segment, bud_break_prob, thickness, thickness_increment, growth_length,\
#                max_length, tie_axis, bud_break_max_length, order, bud_break_prob_func)
    
class Wire():
  """ Defines a trellis wire in the 3D space """
  def __init__(self, id:int, point: tuple, axis: tuple):
    self.__id = id
    self.__axis = axis
    x,y,z = point
    self.point = Vector3(x,y,z)
    self.num_branch = 0
  
  def add_branch(self):
    self.num_branch+=1 
  
class Support():
  """ All the details needed to figure out how the support is structured in the environment, it is a collection of wires"""
  def __init__(self, points: list, num_wires: int, spacing_wires: int, trunk_wire_pt: tuple,\
                branch_axis: tuple, trunk_axis: tuple):
                  
    self.num_wires = num_wires
    self.spacing_wires = spacing_wires
    self.branch_axis = branch_axis
    self.branch_supports = self.make_support(points)#Dictionary id:points
    self.trunk_axis = None
    self.trunk_wire = None
    if trunk_axis:
      self.trunk_axis = trunk_axis
      self.trunk_wire = Wire(-1, trunk_wire_pt, self.trunk_axis) #Make it a vector?
      points.append(trunk_wire_pt)
      
    self.attractor_grid = Point3Grid((1,1,1),list(points))
    
    
  def make_support(self, points):
    supports = {}
    for id,pt in enumerate(points):
      supports[id] = Wire(id, pt, self.branch_axis)
    return supports
      
