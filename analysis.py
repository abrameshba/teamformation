from scipy import stats
import pandas as pd


for network in ['vldb']:
    cmopsodf = pd.DataFrame(columns=['task_size', 'processing_time', 'cardinality', 'diameter', 'leader_dist',
                                'sum_dist', 'gamma', 'shannon', 'inversimpson'])
    blend_df = pd.DataFrame(columns=['task_size', 'processing_time', 'cardinality', 'diameter', 'leader_dist',
                                'sum_dist', 'gamma', 'shannon', 'inversimpson'])
    nsga3df = pd.DataFrame(columns=['task_size', 'processing_time', 'cardinality', 'diameter', 'leader_dist',
                                'sum_dist', 'gamma', 'shannon', 'inversimpson'])
    pplrt_df = pd.DataFrame(columns=['task_size', 'processing_time', 'cardinality', 'diameter', 'leader_dist',
                                'sum_dist', 'gamma', 'shannon', 'inversimpson'])
    hybrid_df = pd.DataFrame(columns=['task_size', 'processing_time', 'cardinality', 'diameter', 'leader_dist',
                                'sum_dist', 'gamma', 'shannon', 'inversimpson'])
    with open("/home/ramesh/dblp/output/" + network + "_cmopso_analysis.txt", "r") as file:
        for line in file:
            vals = line.strip("\n").split("\t")
            values = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])
                      , float(vals[6]), float(vals[7]), float(vals[8])/float(vals[0])]
            cmopsodf.loc[len(cmopsodf)] = values
    with open("/home/ramesh/dblp/output/" + network + "_blend_analysis.txt", "r") as file:
        for line in file:
            vals = line.strip("\n").split("\t")
            values = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])
                      , float(vals[6]), float(vals[7]), float(vals[8])/float(vals[0])]
            blend_df.loc[len(blend_df)] = values
    with open("/home/ramesh/dblp/output/" + network + "_nsga3_analysis.txt", "r") as file:
        for line in file:
            vals = line.strip("\n").split("\t")
            values = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])
                      , float(vals[6]), float(vals[7]), float(vals[8])/float(vals[0])]
            nsga3df.loc[len(nsga3df)] = values
    with open("/home/ramesh/dblp/output/" + network + "_popular_analysis.txt", "r") as file:
        for line in file:
            vals = line.strip("\n").split("\t")
            values = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])
                      , float(vals[6]), float(vals[7]), float(vals[8])/float(vals[0])]
            pplrt_df.loc[len(pplrt_df)] = values
    with open("/home/ramesh/dblp/output/" + network + "_hybrid_analysis.txt", "r") as file:
        for line in file:
            vals = line.strip("\n").split("\t")
            values = [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])
                      , float(vals[6]), float(vals[7]), float(vals[8])/float(vals[0])]
            hybrid_df.loc[len(hybrid_df)] = values
    for cc in ['diameter','leader_dist', 'sum_dist']:
        for dvrst in ['gamma', 'shannon', 'inversimpson']:
            # Perform the two-sample KS test
            print(network,dvrst,cc)
            alpha = 0.05
            # print(anderson(cmopsodf[dvrst] / cmopsodf[cc], dist='norm'))
            # print(anderson(blend_df[dvrst] / blend_df[cc], dist='norm'))
            # print(anderson(nsga3df[dvrst] / nsga3df[cc], dist='norm'))
            # print(anderson(pplrt_df[dvrst] / pplrt_df[cc], dist='norm'))

            # spr_statistic, spr_p_value  = stats.shapiro(cmopsodf[dvrst] / cmopsodf[cc])
            # print(f"Shapiro-Wilk Statistic: {spr_statistic}",f"P-value: {spr_p_value}")
            # spr_statistic, spr_p_value  = stats.shapiro(blend_df[dvrst] / blend_df[cc])
            # print(f"Shapiro-Wilk Statistic: {spr_statistic}",f"P-value: {spr_p_value}")
            # spr_statistic, spr_p_value  = stats.shapiro(nsga3df[dvrst] / nsga3df[cc])
            # print(f"Shapiro-Wilk Statistic: {spr_statistic}",f"P-value: {spr_p_value}")
            # spr_statistic, spr_p_value  = stats.shapiro(pplrt_df[dvrst] / pplrt_df[cc])
            # print(f"Shapiro-Wilk Statistic: {spr_statistic}",f"P-value: {spr_p_value}")
            # if spr_p_value <= alpha:
            #     print("The data does not appear to be normally distributed (reject null hypothesis)")
            # else:
            #     print("The data appears to be normally distributed (fail to reject null hypothesis)")

            ks_statistic, p_value = stats.ks_2samp(cmopsodf[dvrst]/cmopsodf[cc], blend_df[dvrst]/blend_df[cc])
            t_statistic, p_val = stats.ttest_ind(cmopsodf[dvrst]/cmopsodf[cc], blend_df[dvrst]/blend_df[cc])
            print(f"{ks_statistic:.2f}", f"{p_value:.2e}", f"{t_statistic:.2f}", f"{p_val:.2e}")
            ks_statistic, p_value = stats.ks_2samp(nsga3df[dvrst]/nsga3df[cc], blend_df[dvrst]/blend_df[cc])
            t_statistic, p_val = stats.ttest_ind(nsga3df[dvrst]/nsga3df[cc], blend_df[dvrst]/blend_df[cc])
            print(f"{ks_statistic:.2f}", f"{p_value:.2e}", f"{t_statistic:.2f}", f"{p_val:.2e}")
            ks_statistic, p_value = stats.ks_2samp(cmopsodf[dvrst]/cmopsodf[cc], pplrt_df[dvrst]/pplrt_df[cc])
            t_statistic, p_val = stats.ttest_ind(cmopsodf[dvrst]/cmopsodf[cc], pplrt_df[dvrst]/pplrt_df[cc])
            print(f"{ks_statistic:.2f}", f"{p_value:.2e}", f"{t_statistic:.2f}", f"{p_val:.2e}")
            ks_statistic, p_value = stats.ks_2samp(nsga3df[dvrst]/nsga3df[cc], pplrt_df[dvrst]/pplrt_df[cc])
            t_statistic, p_val = stats.ttest_ind(nsga3df[dvrst]/nsga3df[cc], pplrt_df[dvrst]/pplrt_df[cc])
            print(f"{ks_statistic:.2f}", f"{p_value:.2e}", f"{t_statistic:.2f}", f"{p_val:.2e}")
            ks_statistic, p_value = stats.ks_2samp(cmopsodf[dvrst]/cmopsodf[cc], hybrid_df[dvrst]/hybrid_df[cc])
            t_statistic, p_val = stats.ttest_ind(cmopsodf[dvrst]/cmopsodf[cc], hybrid_df[dvrst]/hybrid_df[cc])
            print(f"{ks_statistic:.2f}", f"{p_value:.2e}", f"{t_statistic:.2f}", f"{p_val:.2e}")
            ks_statistic, p_value = stats.ks_2samp(nsga3df[dvrst]/nsga3df[cc], hybrid_df[dvrst]/hybrid_df[cc])
            t_statistic, p_val = stats.ttest_ind(nsga3df[dvrst]/nsga3df[cc], hybrid_df[dvrst]/hybrid_df[cc])
            print(f"{ks_statistic:.2f}", f"{p_value:.2e}", f"{t_statistic:.2f}", f"{p_val:.2e}")
            # Interpret the results
            # if p_value < alpha:
            #     print("Reject the null hypothesis: The two samples are likely from different distributions.")
            # else:
            #     print("Fail to reject the null hypothesis: The two samples may come from the same distribution.")

            # if p_val < alpha:
            #     print("Reject the null hypothesis: There is a significant difference between the two groups.")
            # else:
            #     print("Fail to reject the null hypothesis: There is no significant difference between the two groups.")
