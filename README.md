# Rahuls-Invented-Holistic-Classifier
I invented a ML binary classification algorithm during my sophomore year of college. It was created with the goal of reducing the amount of training data needed to perform accurate classification.

The premise of this algorithm is to treat each "class" as a singular entity. More specifically, the geometric shape of each class (Ex: the geometric shape of the cluster of points from a single class when plotted on a cartesian plane), as well as the relative positions of each class on the cartesian plane are the 2 defining characteristics that drives the generation of the boundary curve. Each class is represented as a peicewise function of its concave hull, thus rendering any interior data point useless. 

This concave hull representation of each class effectively reduces your dataset by >90%, as points that dont lie near the outside of the cluster are ignored. This technique accelerates classification IMMENSELY. The boundary curve is generated in a single pass, and classification is a matter of determining which side of the boundary curve your point lies. 

There are 2 versions, one for 2D datasets and one for N-dimensional space. 

Hyperplane generation does not scale well for N-dimensional datasets; the math is out of my realm. More work must be done to optimize it further. 

To run: 

For 2D version: "python3 holistic_classifier.py"

For the N-D version: "python3 holistic_classifier_scaled.py"

