# File context and how to use them
Every file's context and usage will be explained briefly.

---


## compute_slope_curvature.ipynb
context

**Useage (Jonas)**
Hier komt de uitleg over hoe de code werkt. Denk aan: welk file format gaat hierin, zijn er parameters die je zelf kan instellen?, wat gebeurt er doorheen de code, etc.

---



## curvature_via_irls_quadric_1.ipynb
The IRLS method works by fitting a quadratic plane iteratively to a pointcloud. It starts with a random plane in the pointcloud, then the residuals between that plane and the points are calculated before a new plane is fitted. That iterates a certain amount of times until the best plane with the smallest residuals is fitted.

**Useage**
The full pipeline is explained within the jupyter notebook.
Outputs of the quadratic plane fitting looks something like this:
![IRLS Quadratic Plan Fitting](Pictures/Calculation_via_IRLS.png)

After the plane is fitted, further calculations are done. An example of the output:
![Calculation via IRLS](Pictures/Calculation_via_IRLS.png)


---






