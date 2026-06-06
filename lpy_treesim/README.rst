============
TreeSim_Lpy
============

Description
-----------

TreeSim_Lpy is a tree modelling tool which is built upon L-py with the added features of pruning and tying trees down to mimic different architectures. The tool uses python and prior knowledge of L-systems and L-Py is needed to work with this tool. This tool is ideal for researchers and developers working on botanical simulations and robotic harvesting applications.


Python version 3.9

Table of Contents
-----------------

- `Installation <#installation>`__
- `Tutorials <#tutorials>`__
- `Usage <#usage>`__
- `Features <#features>`__
- `Contributing <#contributing>`__
- `License <#license>`__
- `Contact <#contact>`__
- `Acknowledgments <#acknowledgments>`__

Documentation
-------------

The documentation is provided at `Read the Docs <https://treesim-lpy.readthedocs.io/en/latest/>`__.

You can find the latest L-Py documentation at <https://lpy.readthedocs.io/en/latest>


Installation
------------

To install TreeSim_Lpy, follow these steps (adapted from the `L-Py documentation <https://treesim-lpy.readthedocs.io/en/latest/installation.html>`__):

1. **Install Conda**:
   
   The L-Py distribution relies on the conda software environment management system. If you do not already have conda installed, you can find installation instructions on the `Conda Installation Page <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__.

2. **Create a Conda Environment**:

   Create an environment named `lpy`:
   
   .. code-block:: sh

      conda create -n lpy python=3.9.13 openalea.lpy -c fredboudon -c conda-forge

   The package is retrieved from the `fredboudon` channel (development), and its dependencies will be taken from the `conda-forge` channel.

3. **Activate the L-Py Environment**:

   .. code-block:: sh

      conda activate lpy

4. **Install Required Packages**:

   .. code-block:: sh

      pip install -r requirements.txt

5. **Run L-Py**:

   .. code-block:: sh

      lpy


Tutorials
---------

There are many things you may want to modify as you grow your own trees. Here are some tutorials for some of the more common changes:

1. **Changing Apple Geometry:**
   
   The call production of the apples happens in the ``grow_object(o)`` section:

   .. code-block:: python

       elif 'Apple' in o.name:
           produce [S(.1/2, .09/15)f(.1)&(180)A(.1, .09)]
   
   

   The apple's base is generated with the ``A(bh, r)`` production rule seen below. 

   .. code-block:: python
      
      A(bh, r):
          curves = make_apple_curve()
          base_curve = curves[0]
          top_curve = curves[1]
          nproduce SetColor(230,0,0) SectionResolution(60)
          produce nF(bh, .01, r, base_curve) ^(180) nF(bh/5, .1, r, top_curve)^(180)
   
   The parameters represent the base height of the apple and the radius of the apple. If you wanted to create a completely new apple geometry, just replace the code in this A section. However, if you simply want to edit the existing shape of the apple, that can be done in the ``make_apple_curve()`` section. 

   The apple is made with two curves: a curve that marks the base of the apple, and a curve that marks the indentation on top of the apple. These curves are generated as different Curve2D objects, then turned into QuantisedFunction objects. This is necessary because of the way the apple is produced ``nF``. ``nF`` has an optional parameter ``radiusvariation`` which must be a quantized function. ``nF`` produces a cylinder in n steps, and these curves work by specifying how large the radius for the cylinder should be at each step.
   
   Currently, the stem is produced separately from the apple base. The stem is created in a slightly different way than the apple. A NurbsCurve2D object is returned from the ``make_stem_curve()`` function. This curve is used in ``SetGuide`` to mark how the stem will be generated. ``nF`` is used to follow the guide while generating a cylinder, and there is no ``radiusvariation`` this time.

   .. code-block:: python

       S(sh,r):
           stem_curve = make_stem_curve()
           nproduce SetColor(100,65,23) 
           produce  SetGuide(stem_curve, sh) _(r)nF(sh, .1, r)


2. **Changing Leaf Geometry:**

   The call production of the leaves happens in the ``grow_object(o)`` section:
   
   .. code-block:: python

       elif 'Leaf' in o.name:
           produce L(.1)

   Here, .1 is just a hard-coded value that doesn't have much significance. 

   The actual generation of the leaf can be seen in the ``L(l)`` production section:

   .. code-block:: python

      L(l):
          nproduce SetColor(0,225,0) 
  
          curves = make_leaf_guide()
          curve1 = curves[0]
          curve2 = curves[1]
          produce _(.0025) F(l/10){[SetGuide(curve1, l) _(.001).nF(l, .01)][SetGuide(curve2, l)_(.001).nF(l, .01)]}
   
   The parameter here serves as the length of the leaf. To edit the shape of the leaf, edits can be made in the ``make_leaf_guide()`` function. In the ``make_leaf_guide()`` function, a BezierCurve2D and its inverse are generated. These are returned to be used as guides for the ``SetGuide`` function provided by L-Py. These curves are generated with random points from a set range in order to create leaves of different shape. These randomness of these points, or the range they are generated from, could be edited to change the leaf shape. However, a new geometry altogether could be made in the ``L(l)`` section.
  
          


3. **Changing Bud Geometry:**

   The call for the production of the buds occurs in the ``grow_one(o)`` section by addicting ``spiked_bud(o.thickness)`` to the ``produce`` call: 
   
   .. code-block:: python

      if 'Spur' in o.name:
          produce I(o.growth_length, o.thickness, o) bud(ParameterSet(type=o, num_buds=0)) spiked_bud(o.thickness)grow_object(o)
   
   The actual bud is produced in the ``spiked_bud`` production section: 
   
   .. code-block:: python

      spiked_bud(r):
          base_height = r * 2
          top_height = r * 2
          num_sect = 20
          produce @g(Cylinder(r,base_height,1,num_sect))f(base_height)@g(Cone(r,top_height,0,num_sect))
   
   This is one of the most basic objects generated on the tree model. As the buds on actual trees are little spikes, the bud geometry is made up of a cone on top of a cylinder. These are both produced with L-Py's basic ``@g`` primitive used to draw PglShapes. The height of the two shapes scale with the radius (provided as the parameter). The ``num_sect`` is used to determine how many sections each shape is made up of, and 20 was chosen as they appear circular without adding too many triangles to the model. 
   
   

4. **Changing Branch Profile Curve:**
   
   Every branch on the model has a unique profile curve. This is so the model doesn't appear to be perfectly cylindrical (as if made of PVC pipes). Every branch has its own unique curve that is generated when the branch is originally generated. This can be found in the actual class declaration at the beginning of the code:
   
   .. code-block:: python
   
      self.contour = create_noisy_circle_curve(1, .2, 30)
   
   Every ``Trunk``, ``Branch``, and ``NonBranch`` object have a their own unique curve associated with them. However this curve is only applied in the ``grow_one(o)`` section:
   
   .. code-block:: python

      if 'Trunk' in o.name or 'Branch' in o.name:
          nproduce SetContour(o.contour)
      else:
          reset_contour()
   
   This code targets every ``Trunk``, ``Branch``, and ``NonBranch`` object and sets their own curve when growing them. Whenever any other object is passed to the ``grow_object(o)`` function the call to ``reset_contour()`` sets the contour back to a perfect circle to ensure that no curves overlap. 
   
   The ``create_noisy_circle()`` function is included in the helper.py file. It works by creating a circle out of a given number of points, and then moving those points in the x and y direction according to a given noise factor. The function has two required parameters and two optional parameters. The two required parameters are ``radius`` and ``noise_factor``. The ``radius`` determines the radius of the generated circle. The ``noise_factor`` is used to set a range in which random points will be generated. The points making up the circle will then be moved a random amount in that range. This can be seen in the main ``for`` loop in the function:
   
   .. code-block:: python
   
      for angle in t:
          # Base circle points
          x = radius * cos(angle)
          y = radius * sin(angle)
      
          # Add noise
          noise_x = uniform(-noise_factor, noise_factor)
          noise_y = uniform(-noise_factor, noise_factor)
      
          noisy_x = x + noise_x
          noisy_y = y + noise_y
      
          points.append((noisy_x, noisy_y))
   
   The two optional parameters for the function are ``num_points`` and ``seed``. ``num_points`` is used to determine how many points make up the circle. If no value is given, it defaults to 100. ``seed`` is used to set the randomness of the circles. If a value is given, the random.seed is set to that value. For this model, seeds were not used.  

5. **Changing Tertiary Branch Curves:**
   
   Every tertiary branch on the model follows a unique curve. This curve is generated with the ``create_bezier_curve()`` function which can be found in the helper.py file. The function works by generating four points to be used as guide points for the Bézier curve. There is some control code to make sure that the x points are random but are still generated in a linear fashion. 
   
   The function takes four optional parameters: ``num_control_points``, ``x_range``, ``y_range``, and ``seed_val``. ``num_control_points`` defaults to four, the standard for Bézier curves. ``x_range`` and ``y_range`` default to a tuples designating the ranges: (0,10) and (-2,2) respectively. ``seed_val`` defaults to ``None``, however in the actual model ``time.time()`` is used as the seed. 

   The actual designation of the curves for the tertiary branches can be seen in the ``bud(t)`` section:
   
   .. code-block:: python

      if 'NonTrunk' in new_object.name:
          import time
          curve = create_bezier_curve(seed_val=time.time())
          nproduce [SetGuide(curve, new_object.max_length)
   
   A curve is generated with the ``create_bezier_curve()`` and it is used as the guide for the ``SetGuide`` function provided by L-Py. 

6. **Changing color ID system:**
   
   Every object on the tree has its own unique color code to act as an ID. The implementation of this can be seen in the ``grow_object(o)`` section:

   .. code-block:: python

      r, g, b = o.color
      nproduce SetColor(r, g, b)
      
      smallest_color = [r, g, b].index(min([r, g, b]))
      o.color[smallest_color] += 1
   
   This works by finding the smallest value of the objects RGB code and incrementing it by one. This is to avoid any kind of error where a color value is greater than 255. To get rid of the color IDs, simply comment out the second two lines of the code above. 
   
   

7. **Changing Apple and Leaf ratio:**
   
   The generation of apples and leaves is random. If something is to grow off of the bud, there is a 90% chance it will be a leaf, and a 10% chance it will be an apple. These percentages are set in the ``Spur`` class declaration: 
      
   .. code-block:: python

      # From Spur class
      def create_branch(self):
          if self.num_leaves < self.max_leaves: 
              self.num_leaves += 1
              if rd.random()<0.9:
                  new_ob = Leaf(copy_from = self.prototype_dict['leaf'])
              else:
                  new_ob = Apple(copy_from = self.prototype_dict['apple'])
          else: 
              new_ob = None
          
          return new_ob
   
   


Features
--------

- **Pruning:** Remove unwanted branches to simulate pruning.
- **Branch Tying:** Simulate branches being tied down to mimic different orchard architectures.
- **Model Class Types:** The model generated is built with classes of different material type. 

========
Gallery
========
.. figure:: media/envy_model.png
   :width: 500
   :height: 500
   
   Example of a labelled, pruned and tied envy tree system using TreeSim_Lpy
  
  

.. figure:: media/ufo.png
   :width: 500
   :height: 300
   
   Example of a labelled, pruned and tied UFO tree system using TreeSim_Lpy


Contact
-------

For any questions or issues, please contact us through **GitHub Issues**. 


Help and Support
----------------

Please open an **Issue** if you need support or you run into any error (Installation, Runtime, etc.).
We'll try to resolve it as soon as possible.


==============
Citations
==============

   - F. Boudon, T. Cokelaer, C. Pradal, P. Prusinkiewicz and C. Godin, L-Py: an L-system simulation framework for modeling plant architecture development based on a dynamic language, Frontiers in Plant Science, 30 May 2012.

