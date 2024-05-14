import train_MLP as tfc
import train_conv as tconv
import train_convImgN as tconvIm
import train_AE as tae
import wt_analyze as wtAnlz
import err_analyze as errAnlz
import time_analyze as tmAnlz
import time_analyze_conv as tmAnlzC
import time_analyze_ae as tmAnlzAE
import infer_compare as infComp
import proximal_analyze as proxAnz
import plot
import train_CIFAR_100_fc as tcf_100
import train_CIFAR_100_conv as tcf_100_conv
import train_Caltech256 as tct_256_conv
import train_labelme as tlm


def main():
    ####################### CLASSIFICATION FULLY CONNECTED #############################

    # # SVHN
    # # BP
    # print("done")
    # tfc.training_run(epochs=70, data=0, num_seeds=5, alpha=[.019], model_type=0)
    # print("done")
    # # BP-Adam
    # tfc.training_run(epochs=70, data=0, num_seeds=5, alpha=[.000045], model_type=1)
    # # IL
    # tfc.training_run(epochs=70, data=0, num_seeds=5, beta=100, model_type=2, alpha=[1])
    # # IL-MQ
    # tfc.training_run(epochs=70, data=0, num_seeds=5, beta=100, model_type=3, alpha=[.00003])
    # # IL-Adam
    # tfc.training_run(epochs=70, data=0, num_seeds=5, beta=100, model_type=4, alpha=[.00003])


    # # Cifar-10
    # # BP
    # tfc.training_run(epochs=110, data=1, num_seeds=5, alpha=[.01], model_type=0)
    # # BP-Adam
    # tfc.training_run(epochs=110, data=1, num_seeds=5, alpha=[.000018], model_type=1)
    # # IL
    # tfc.training_run(epochs=110, data=1, num_seeds=5, beta=100, model_type=2, alpha=[.75])
    # # IL-MQ
    # tfc.training_run(epochs=110, data=1, num_seeds=5, beta=100, model_type=3, alpha=[.00003])
    # # IL-Adam
    # tfc.training_run(epochs=110, data=1, num_seeds=5, beta=100, model_type=4, alpha=[.00003])



    # ####################### CLASSIFICATION CONVOLUTIONAL #############################
    # # SVHN
    # # BP
    # tconv.training_run(epochs=45, data=0, num_seeds=5, alpha=[.01], model_type=0)
    # # BP-Adam
    # tconv.training_run(epochs=45, data=0, num_seeds=5, alpha=[.0001], model_type=1)
    # # IL
    # tconv.training_run(epochs=45, data=0, num_seeds=5, beta=100, model_type=2, alpha=[.25])
    # # IL-MQ
    # tconv.training_run(epochs=45, data=0, num_seeds=5, beta=100, model_type=3, alpha=[.00007])
    # # IL-Adam
    # tconv.training_run(epochs=45, data=0, num_seeds=5, beta=100, model_type=4, alpha=[.00001])


    # # CIFAR-10
    # # BP
    # tconv.training_run(epochs=45, data=1, num_seeds=5, alpha=[.014], model_type=0)
    # # BP-Adam
    # tconv.training_run(epochs=45, data=1, num_seeds=5, alpha=[.00012], model_type=1)
    # # IL
    # tconv.training_run(epochs=45, data=1, num_seeds=5, beta=100, model_type=2, alpha=[.5])
    # # IL-MQ
    # tconv.training_run(epochs=45, data=1, num_seeds=5, beta=100, model_type=3, alpha=[.00022])
    # # IL-Adam
    # tconv.training_run(epochs=45, data=1, num_seeds=5, beta=100, model_type=4, alpha=[.000075])


    # # Tiny ImageNet
    # # BP
    # tconvIm.training_run(epochs=20, num_seeds=5, alpha=[.05], model_type=0)
    # # BP-Adam
    # tconvIm.training_run(epochs=20, num_seeds=5, alpha=[.00025], model_type=1)
    # # IL
    # tconvIm.training_run(epochs=20, num_seeds=5, beta=100, model_type=2, alpha=[.3])
    # # IL-MQ
    # tconvIm.training_run(epochs=20, num_seeds=5, beta=100, model_type=3, alpha=[.00012])
    # # IL-Adam
    # tconvIm.training_run(epochs=20, num_seeds=5, beta=100, model_type=4, alpha=[.00007])

    # CIFAR-100

    # h-layers = 5
    # BP
    # print("BP")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, alpha=[.014], model_type=0, n_hlayers=5)
    # # BP-Adam
    # print("Adam")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, alpha=[.00012], model_type=1, n_hlayers = 5)
    # # IL
    # print("IL")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, beta=100, model_type=2, alpha=[.5], n_hlayers = 5)
    # # IL-MQ
    # print("IL-MQ")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, beta=100, model_type=3, alpha=[.00022], n_hlayers = 5)
    # # IL-Adam
    # print("IL-Adam")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, beta=100, model_type=4, alpha=[.000075], n_hlayers = 5)
    # print("Done")

    # # h-layers = 10
    # # BP
    # print("BP")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, alpha=[.014], model_type=0, n_hlayers = 10)
    # # BP-Adam
    # print("Adam")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, alpha=[.00012], model_type=1, n_hlayers = 10)
    # # IL
    # print("IL")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, beta=100, model_type=2, alpha=[.5], n_hlayers = 10)
    # # IL-MQ
    # print("IL-MQ")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, beta=100, model_type=3, alpha=[.00022], n_hlayers = 10)
    # # IL-Adam
    # print("IL-Adam")
    # tcf_100.training_run(epochs=30, data=5, num_seeds=1, beta=100, model_type=4, alpha=[.000075], n_hlayers = 10)
    # print("Done")

    # # h-layers = 15
    # # BP
    # print("BP")
    # tcf_100.training_run(epochs=45, data=5, num_seeds=1, alpha=[.014], model_type=0, n_hlayers = 15)
    # # BP-Adam
    # print("Adam")
    # tcf_100.training_run(epochs=45, data=5, num_seeds=1, alpha=[.00012], model_type=1, n_hlayers = 15)
    # # IL
    # print("IL")
    # tcf_100.training_run(epochs=45, data=5, num_seeds=1, beta=100, model_type=2, alpha=[.5], n_hlayers = 15)
    # # IL-MQ
    # print("IL-MQ")
    # tcf_100.training_run(epochs=45, data=5, num_seeds=1, beta=100, model_type=3, alpha=[.00022], n_hlayers = 15)
    # # IL-Adam
    # print("IL-Adam")
    # tcf_100.training_run(epochs=45, data=5, num_seeds=1, beta=100, model_type=4, alpha=[.000075], n_hlayers = 15)
    # print("Done")

    # LabelMe50K

    # # small
    # # BP
    # print("BP")
    # tlm.training_run(epochs=20, num_seeds=1, alpha=[.014], model_type=0, small = True)
    # # BP-Adam
    # print("Adam")
    # tlm.training_run(epochs=20, num_seeds=1, alpha=[.00012], model_type=1, small = True)
    # # IL
    # print("IL")
    # tlm.training_run(epochs=20, num_seeds=1, beta=100, model_type=2, alpha=[.5], small = True)
    # # IL-MQ
    # print("IL-MQ")
    # tlm.training_run(epochs=20, num_seeds=1, beta=100, model_type=3, alpha=[.00022], small = True)
    # # IL-Adam
    # print("IL-Adam")
    # tlm.training_run(epochs=20, num_seeds=1, beta=100, model_type=4, alpha=[.000075], small = True)
    # print("Done")

    # # big
    # # BP
    # print("BP")
    # tlm.training_run(epochs=20, num_seeds=1, alpha=[.014], model_type=0, small = False)
    # # BP-Adam
    # print("Adam")
    # tlm.training_run(epochs=20, num_seeds=1, alpha=[.00012], model_type=1, small = False)
    # # IL
    # print("IL")
    # tlm.training_run(epochs=20, num_seeds=1, beta=100, model_type=2, alpha=[.5], small = False)
    # # IL-MQ
    # print("IL-MQ")
    # tlm.training_run(epochs=20, num_seeds=1, beta=100, model_type=3, alpha=[.00022], small = False)
    # # IL-Adam
    # print("IL-Adam")
    # tlm.training_run(epochs=20, num_seeds=1, beta=100, model_type=4, alpha=[.000075], small = False)
    # print("Done")


    # CIFAR100 - COnv

    # n_iters = [10, 15]
    # batch_sizes = [32, 64, 128]

    # n_iters = [3]
    # batch_sizes = [64]
    epochs = 15

    # for n_iter in n_iters:
    #     for batch_size in batch_sizes:
    #         # BP
    #         print("BP")
    #         tcf_100_conv.training_run(epochs=epochs, num_seeds=1, alpha=[.014], model_type=0, n_iter=n_iter, batch_size=batch_size)
    #         # BP-Adam
    #         print("Adam")
    #         tcf_100_conv.training_run(epochs=epochs, num_seeds=1, alpha=[.00012], model_type=1, n_iter=n_iter, batch_size=batch_size)
    #         # IL
    #         print("IL")
    #         tcf_100_conv.training_run(epochs=epochs, num_seeds=1, beta=100, model_type=2, alpha=[.5], n_iter=n_iter, batch_size=batch_size)
    #         # IL-MQ
    #         print("IL-MQ")
    #         tcf_100_conv.training_run(epochs=epochs, num_seeds=1, beta=100, model_type=3, alpha=[.00022], n_iter=n_iter, batch_size=batch_size)
    #         # IL-Adam
    #         print("IL-Adam")
    #         tcf_100_conv.training_run(epochs=epochs, num_seeds=1, beta=100, model_type=4, alpha=[.000075], n_iter=n_iter, batch_size=batch_size)
    #         print("Done")

    n_iters = [3]
    batch_sizes = [256]
    smalls = [True, False]

    for small in smalls:
        for batch_size in batch_sizes:
            for n_iter in n_iters:
                # print("BP")
                # tct_256_conv.training_run(epochs=epochs, num_seeds=1, alpha=[.014], model_type=0, n_iter=n_iter, batch_size=batch_size, small=small)
                # BP-Adam
                # print("Adam")
                # tct_256_conv.training_run(epochs=epochs, num_seeds=1, alpha=[.00012], model_type=1, n_iter=n_iter, batch_size=batch_size, small=small)
                # IL
                # print("IL")
                # tct_256_conv.training_run(epochs=epochs, num_seeds=1, beta=100, model_type=2, alpha=[.5], n_iter=n_iter, batch_size=batch_size, small=small)
                # IL-MQ
                print("IL-MQ")
                tct_256_conv.training_run(epochs=epochs, num_seeds=1, beta=100, model_type=3, alpha=[.001], n_iter=n_iter, batch_size=batch_size, small=small)
                # IL-Adam
                # print("IL-Adam")
                # tct_256_conv.training_run(epochs=epochs, num_seeds=1, beta=100, model_type=4, alpha=[.000075], n_iter=n_iter, batch_size=batch_size, small=small)
                print("Done")
    

    # ################################## AUTOENCODER #####################################
    # # SVHN
    # # BP
    # tae.training_run(epochs=100, data=0, num_seeds=5, alpha=[.0013], model_type=0)
    # # BP-Adam
    # tae.training_run(epochs=100, data=0, num_seeds=5, alpha=[.0009], model_type=1)
    # # IL
    # tae.training_run(epochs=100, data=0, num_seeds=5, beta=100, model_type=2, alpha=[.05])
    # # IL-MQ
    # tae.training_run(epochs=100, data=0, num_seeds=5, beta=100, model_type=3, alpha=[.000016])
    # # IL-Adam
    # tae.training_run(epochs=100, data=0, num_seeds=5, beta=100, model_type=4, alpha=[.000045])

    # # CIFAR-10
    # # BP
    # tae.training_run(epochs=100, data=1, num_seeds=5, alpha=[.0013], model_type=0)
    # # BP-Adam
    # tae.training_run(epochs=100, data=1, num_seeds=5, alpha=[.0007], model_type=1)
    # # IL
    # tae.training_run(epochs=100, data=1, num_seeds=5, beta=100, model_type=2, alpha=[.1])
    # # IL-MQ
    # tae.training_run(epochs=100, data=1, num_seeds=5, beta=100, model_type=3, alpha=[.000012])
    # # IL-Adam
    # tae.training_run(epochs=100, data=1, num_seeds=5, beta=100, model_type=4, alpha=[.00004])


    ########################## WEIGHT UPDATE MAGNITUDE + ERROR ANALYSIS ########################
    # for m in range(3,6):
    #     wtAnlz.training_run(model_type=m)
    # errAnlz.training_run()
    # print("After errAnlz")


    ################################## TIME ANALYSIS #########################################
    #Get training times
    # for m in [3]:
    #     tmAnlz.training_run(epochs=10, data=1, model_type=m)
    #     tmAnlzC.training_run(epochs=10, data=1, model_type=m)
    #     tmAnlzAE.training_run(epochs=10, data=1, model_type=m)
    # print("finish time analysis")

    ################################ Inference Compare ##################################
    # Grid search: alphas=[.0001, .00005, .00001]  gammas=[.02, .05, .1, .3]
    # seq_alphas = [.0001, .00005, .00005, .00005, .00005]
    # seq_gammas = [.05, .05, .05, .05, .05]
    # sim_alphas = [.0001, .0001, .0001, .00005, .00005]
    # sim_gammas = [.1, .02, .02, .05, .05]
    # for n_iter in range(1, 6):
    #     print(f'# Inference Iterations:{n_iter}')
    #     print('Sequential IL')
    #     infComp.training_run(epochs=45, data=1, num_seeds=5, model_type=0, n_iter=n_iter,
    #                          alpha=[seq_alphas[n_iter-1]], gamma=[seq_gammas[n_iter-1]])

    #     print('Standard/Simultaneous IL')
    #     infComp.training_run(epochs=45, data=1, num_seeds=5, model_type=1, n_iter=n_iter,
    #                          alpha=[sim_alphas[n_iter-1]], gamma=[sim_gammas[n_iter-1]])
    
    # print("finish inference compare")


    ############################### Proximal Analyze ##################################
    # proxAnz.training_run(num_seeds=1, model_type=1, alpha=.1, gamma=.01, opt_type=0, test_iter=1)
    # proxAnz.training_run(num_seeds=1, model_type=1, alpha=.00003, gamma=.01, opt_type=1, test_iter=1)
    # proxAnz.training_run(num_seeds=1, model_type=0, alpha=1, gamma=.04, opt_type=0, n_iter=5, test_iter=1)
    # proxAnz.training_run(num_seeds=1, model_type=0, alpha=.00003, gamma=.04, opt_type=1, n_iter=5, test_iter=1)
    # print("finish prox analyze")



    ################################## Plot ##############################################
    # plot.plot_conv()
    # plot.plot_MLP()
    # plot.plot_T_Analyze_training()
    # plot.plot_wt_anlz()
    # plot.plot_wt_anlz_conv()
    # plot.plot_speed_analysis()
    # plot.plot_T_anlz()
    # plot.plot_prox_Analyze()
    # plot.plot_prox()
    print("finish plot")


main()
