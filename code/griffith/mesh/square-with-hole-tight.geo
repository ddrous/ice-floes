/*
Squared mesh
*/

lc = 2; // mesh precision
Point(1) = {0, 0, 0, lc};
Point(2) = {100, 0, 0, lc};
Point(3) = {100, 100, 0, lc};
Point(4) = {0, 100, 0, lc};
Point(5) = {50, 50, 0, lc};
Point(6) = {45, 50, 0, lc};
Point(7) = {55, 50, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 6};
Physical Line("D") = {2, 4, -5, -6};
Physical Line("N") = {1, 3};
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6};
Plane Surface(1) = {1, 2};
Physical Surface("S") = {1};
