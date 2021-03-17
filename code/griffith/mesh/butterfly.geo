/*
Butterfly mesh
*/

lc = 9; // mesh precision
Point(1) = {0, 0, 0, lc};
Point(2) = {100, 0, 0, lc};
Point(3) = {100, 100, 0, lc};
Point(4) = {0, 100, 0, lc};
Line(1) = {1, 4};
Line(2) = {2, 3};
Physical Line("D") = {-1, 2};
Point(5) = {50, 130, 0, 1.0};
Circle(3) = {4, 5, 3};
Point(6) = {50, -30, 0, 1.0};
Circle(4) = {1, 6, 2};
Line Loop(1) = {3, -2, -4, 1};
Physical Line("N") = {-3, 4};
Plane Surface(1) = {1};
Physical Surface("S") = {1};
