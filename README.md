# Fracture of ice floes by percussion in a granular model

This repository contains files used and developed for the percussion and the fracture of an ice floe (viewed as mass-spring-damper system in 1D and in 2D), following the Griffith model. This repository was created for the purpose of a Master's level internship at the Jacques Louis-Lions Laboratory.

---
## Structure
- `docs` contains useful information to understand the problem at hand. The most important files here are the two theses: Rabatel (2015), and Balasoiu (2020).
- `reports` contains weekly reports during the internship, and the final report.
- `slides` contains a presentation for the Master thesis defense.
- `pdfs` contains the PDF files (final report and slides) submitted for thesis defense.
- `code` contains the 1D and 2D code that was developed. To launch a 1D simulation, adjusts parameters and run the Python scripts `Percussion1D-CLI.py` and `Fracture1D-CLI.py` (this latter script also deals with Percussion). Jupyter notebooks are also available to test specific aspects of the problem.

---
## Screenshots
A few results from simulations are available through [this link](SEAFILE).
<!-- ![Screenshot 1](.\reports\internship\Figures\Screenshot1.jpg)
![Screenshot 2](.\reports\internship\Figures\Screenshot2.jpg)
![Screenshot 3](.\reports\internship\Figures\Screenshot3.jpg)
![Screenshot 4](.\reports\internship\Figures\Screenshot4.jpg)
![Screenshot 5](.\reports\internship\Figures\Screenshot5.jpg) -->
![Screenshot 1](https://github.com/desmond-rn/ice-floes/blob/master/reports/internship/Figures/Screenshot1.jpg)
![Screenshot 2](https://github.com/desmond-rn/ice-floes/blob/master/reports/internship/Figures/Screenshot2.jpg)
![Screenshot 3](https://github.com/desmond-rn/ice-floes/blob/master/reports/internship/Figures/Screenshot3.jpg)
![Screenshot 4](https://github.com/desmond-rn/ice-floes/blob/master/reports/internship/Figures/Screenshot4.jpg)
![Screenshot 5](https://github.com/desmond-rn/ice-floes/blob/master/reports/internship/Figures/Screenshot5.jpg)

---
## Resources
All the classic scientific computing libraries (Numpy, Scipy, etc.) will have to be installed to run the scripts. Additionally, one might need:
- [Bokeh](https://bokeh.org/) for interactive plotting in Notebooks.
- [PILImage](https://pillow.readthedocs.io/en/stable/reference/Image.html) for exporting simulation results as gifs.
<!-- --- -->