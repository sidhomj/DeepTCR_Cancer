# Deep learning reveals predictive sequence concepts within immune repertoires to immunotherapy

TCR-Seq (T-cell receptor sequencing) has been used extensively to characterize the immune response to cancer. However, most analyses of this data have been restricted to assessing quantitative measures such as clonality that do not leverage the information within the CDR3 sequence. We use DeepTCR, a previously described set of deep learning algorithms, to reveal sequence concepts that are predictive of response to immunotherapy in a cohort of clinical trial patients. We demonstrate that a multiple-instance deep learning algorithm can predict response with a high degree of accuracy and further utilize the model to infer the antigenic specificities of the predictive signature and their unique dynamics during therapy. Our results demonstrate that the predictive signature of non-response is associated with high frequencies of TCRs predicted to recognize tumor-specific antigens and these tumor-specific TCRs undergo a higher degree of dynamic changes on therapy in non-responders vs. responders. These results are consistent with a biological model where the hallmark of non-responders is an accumulation of tumor-specific T-cells that undergo turnover on therapy, possibly because of the dysfunctional state of these T-cells in non-responders.

 All scripts used to do all the analyses described in the manuscript for the purpose of methodological transparency. They can be found under the scripts folder and are organized by the type of analysis. All required software that was used in this analysis can be found under requirements.txt.
 
## Publication

For description of analyses and their results contained in this repository, refer to the following manuscript:

[Sidhom, J.W., Oliveria, G., Ross-MacDonald, R., Wind-Rotolo, M., Wu, C.J., Pardoll, D.M., & Baras, A.S. (2022). Deep learning reveals predictive sequence concepts within immune repertoires to immunotherapy. Science Advances - in press]()

For full description of algorithm and methods behind [DeepTCR](https://github.com/sidhomj/DeepTCR), refer to the following manuscript:

[Sidhom, J.W., Larman, H.B., Pardoll, D.M., & Baras, A.S. (2021). DeepTCR is a deep learning framework for revealing sequence concepts within T-cell repertoires. Nat Commun 12, 1605](https://www.nature.com/articles/s41467-021-21879-w)
