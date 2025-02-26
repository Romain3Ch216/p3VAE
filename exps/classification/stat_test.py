from scipy.stats import shapiro, levene, ttest_ind

score_q_no_gs = c(0.8771746001865054,
            0.9034996070969992,
            0.8962500634004125,
            0.8953423959523723,
            0.8791518353673609,
            0.8567377609707422,
            0.9004883197520421,
            0.8834456402441871,
            0.8778121881223464,
            0.8660826117317227)

score_q_gs = c(0.9231048819614895,
            0.8961068970982504,
            0.8990425416311785,
            0.8899071839445779,
            0.9135532041245649,
            0.8770316008910044,
            0.90701326333487,
            0.8915013329550235,
            0.8831735231554051,
            0.9031859306329602)

shapiro_test = shapiro(score_q_no_gs)
print(shapiro_test.pvalue)

shapiro_test = shapiro(score_q_gs)
print(shapiro_test.pvalue)

var_test = levene(score_q_no_gs, score_q_gs)
print(var_test.pvalue)

student_test = ttest_ind(score_q_no_gs, score_q_gs, equal_var=True, alternative='less')
print(student_test.pvalue)
