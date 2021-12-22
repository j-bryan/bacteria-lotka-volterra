# Identification of Lotka-Volterra Dynamics for Bacterial Growth

This is an undergraduate research project, conducted under Geordie Richards and Clara Cho. We modeled the competitive dynamics of in-vitro bacterial growth to investigate how various species might interact in the gut microbiome. To do this, we applied several models:

1. Least-squares regression of Lotka-Volterra (LV) interaction parameters
2. An Extended Kalman Filter (EKF) and augmented state extended Kalman filter (ASEKF) to optimally estimate the LV parameters over time. This approach allows the parameters to change over time.
3. Sparse identification of nonlinear systems (SINDy, see Brunton et al. (2016)). This method attempts to learn a sparse system of differential equations which best fit the data from a library of candidate functions.

Least squares regression provided consistent results, while the EKF was sensitive to observation frequency and SINDy was sensitive to the threshold parameter, making these methods more unreliable in this setting with the available data. It is possible that further data acquisition would alleviate some of this sensitivity to parameter selection. None of these models performed particularly well at predicting future interaction behavior.

## Related Reading
[1]Cho CE, Taesuwan S, Malysheva OV, Bender E, Tulchinsky NF, Yan J, et al. Trimethylamine-N-oxide (TMAO) response to animal source foods varies among healthy young men and is influenced by their gut microbiota composition: A randomized controlled trial. Mol Nutr Food Res. 2017;61.
[2] Ley RE, Backhed F, Turnbaugh P, Lozupone CA, Knight RD, Gordon JI. Obesity alters gut microbial ecology. Proc Natl Acad Sci U S A. 2005;102:11070-5.
[3] X.-Y. Li, C. Pietschke, S. Fraune, P. M. Altrock, T. C. G. Bosch, and A. Traulsen, “Which games are growing bacterial populations playing?,” Journal of The Royal Society Interface, vol. 12, no. 108, p. 20150121, Jul. 2015.
[4] M. Alshawaqfeh, E. Serpedin, and A. B. Younes, “Inferring microbial interaction networks from metagenomic data using SgLV-EKF algorithm,” BMC Genomics, vol. 18, no. S3, Mar. 2017.
[5] R. Chartrand, “Numerical Differentiation of Noisy, Nonsmooth Data,” ISRN Applied Mathematics, vol. 2011, pp. 1–11, 2011.
[6] S. L. Brunton, J. L. Proctor, and J. N. Kutz, “Discovering governing equations from data: Sparse identification of nonlinear dynamical systems,” Proceedings of the National Academy of Sciences, vol. 113, no. 15, pp. 3932–3937, Apr. 2016.
