import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx


params = {'font.family': 'serif',
          'font.serif': 'Times New Roman',
          'font.style': 'normal',
          'font.weight': 'normal',  # or 'blod'
          'figure.figsize': [37, 7],
          # 'font.size': 25
          }
plt.rcParams.update(params)


def plot_combine(result_dir, chi):
    plt.rc('xtick', labelsize=40)
    plt.rc('ytick', labelsize=40)
    plt.rc('legend', fontsize=40)
    plt.rc('lines', lw=5, markersize=15)
    font_title = {'family': 'Verdana',
                  'style': 'normal',
                  'weight': 'normal',
                  'size': 60,
                 }
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(55, 13))
    plt.subplots_adjust(wspace=0.05)
    ax = ax.flatten()
    graph_pool = ['lattice', 'rrg', 'sw', 'complete']
    annotate = ['(a)', '(b)', '(c)', '(d)']
    Jij_pool = ['ones', 'randn', 'randn', 'sk']
    n = [256, 80, 70, 20]
    for i in range(4):
        if i == 0:
            beta = np.arange(0.1, 1.1, 0.1)
            L = 16
            node = 'mps'
            if os.path.exists('results/squarelatticeL16/kacward{}.txt'.format(L)):
                exact = np.loadtxt('results/squarelatticeL16/kacward{}.txt'.format(L))

            # fe_tn = np.loadtxt('{}n256chi{}{}.txt'.format(result_dir, chi, node))
            fe_tn_np = np.loadtxt('results/squarelatticeL16/n256chi{}{}_np.txt'.format(chi, node))
            free_energy = np.loadtxt('results/squarelatticeL16/results.txt')
            fe_MF = np.loadtxt('results/squarelatticeL16/MFn{}.txt'.format(L ** 2))

            # left, bottom, width, height = 0.12, 0.12, 0.8, 0.8
            """
            ax1 = fig.add_axes([left, bottom, width, height])
            ax1.set_xlim((0.301, 1.02))
            ax1.set_ylim((-2.2, -1.85))
            ax1.set_yticks(np.arange(-2.2, -1.9, 0.1))
            ax1.tick_params(labelsize=15)
            ax1.set_xlabel(r'$\beta$', fontsize=22)
            ax1.set_ylabel('Free Energy', fontsize=22)
            ax1.plot(betas[33:], exact[33:], c='k')
            ax1.scatter(beta[3:], free_energy[1][3:], facecolors='none', edgecolors='y', marker='o')
            ax1.scatter(beta[3:], free_energy[3][3:], c='b', marker="x")
            ax1.scatter(beta[3:], free_energy[2][3:], c='tab:orange', marker="d")
            ax1.scatter(beta[3:], free_energy[0][3:], c='tab:cyan', marker='*')
            ax1.scatter(beta[3:], fe_tn[0][3:], c='r')
            ax1.legend(['Exact', 'Bethe', 'Dense', 'Conv', 'FVS', 'TN'], loc=2, ncol=1, fontsize=15, frameon=False)
            """
            # plt.axes([0.55, 0.2, 0.35, 0.45])
            # plt.axes([left, bottom, width, height])
            ax[0].axvline(0.4406868, color='k', linestyle='--', label='_nolegend_')
            """
            plt.scatter(beta, np.log10(np.abs(np.array(fe_tn[0]) - exact[0:100:11])),
                        c='r')
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[1]) - exact[0:100:11])),
                        facecolors='none', edgecolors='y', marker="o")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[2]) - exact[0:100:11])),
                        c='tab:orange', marker="d")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[3]) - exact[0:100:11])),
                        c='b', marker="x")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[0]) - exact[0:100:11])),
                        c='tab:cyan', marker='*')
            """
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[0]) - exact)), c='r', marker='<', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[2]) - exact)), c='r', marker='>', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[4]) - exact)), c='r', marker='^', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[6]) - exact)), c='r', marker='v', mfc='none')
            '''
            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[0]) - exact)),
                       c='r', marker='<', mfc='none', label='Dmax: 1')
            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[2]) - exact)),
                       c='r', marker='>', mfc='none', label='Dmax: 10')
            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[4]) - exact)),
                       c='r', marker='^', mfc='none', label='Dmax: 20')
            '''
            lines = ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[6]) - exact)),
                               c='r', marker='v', mfc='none', label='Our method')
            lines += ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[2]) - exact)),
                                c='c', marker="*", label='Conv VAN')  # Conv
            lines += ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[3]) - exact)),
                                c='k', marker='*', label='VAN')  # Dense
            # plt.plot(beta, np.log10(np.abs(np.array(free_energy[0]) - exact[0:100:11])),
            #          c='tab:cyan', marker='*')  # FVS
            lines += ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[1]) - exact)),
                                c='y', mfc='none', mec='y', marker="o", label='Bethe')  # Bethe
            lines += ax[0].plot(beta, np.log10(np.abs(fe_MF[1] - exact)),
                                c='g', marker="d", label='TAP')  # TAP
            lines += ax[0].plot(beta, np.log10(np.abs(fe_MF[2] - exact)),
                                c='b', marker='x', label='NMF')  # NMF
            lines.reverse()
            ax[0].set_ylim((-18, 0))
            ax[0].set_yticks([-1, -4, -7, -10, -13, -16])
            ax[0].set_yticklabels(['$10^{-1}$', '$10^{-4}$', '$10^{-7}$', '$10^{-10}$', '$10^{-13}$', '$10^{-16}$'],
                                  fontsize=40)
            ax[0].set_xticks(beta[0:10:2])
            # ax[0].set_xticklabels(ax[0].get_xticks(), fontsize=40)
            ax[0].set_ylabel('Relative Error', fontsize=45)
            ax[0].set_xlabel(r'$\beta$', fontsize=45)
            # ax[0].set_title(annotate[0], fontsize=50)
            ax[0].text(0.5, -0.2, annotate[0], fontdict=font_title, transform=ax[0].transAxes, ha="center")
            ax0 = ax[0].inset_axes([0.5, 0.275, 0.4, 0.4])
            L = 5
            grid = nx.grid_2d_graph(L, L)
            pos = list(grid.nodes)
            pos_grid = {}.fromkeys(np.arange(len(pos)))
            for i in pos_grid.keys():
                pos_grid[i] = pos[i]
            edges_2d = list(grid.edges)
            edges = [(i[0] * L + i[1], j[0] * L + j[1]) for i, j in edges_2d]
            graph = nx.Graph()
            graph.add_nodes_from(np.arange(len(pos)))
            graph.add_edges_from(edges)
            nodes = nx.draw_networkx_nodes(graph, pos_grid, ax=ax0)
            nodes.set_color('white')
            nodes.set_edgecolor('k')
            nx.draw_networkx_edges(graph, pos_grid, ax=ax0, width=3)
            ax0.set_axis_off()
            # ax[0].legend(loc='center left', ncol=2, fontsize=20, frameon=False)
        else:
            D = 16
            beta = np.arange(0.1, 2.1, 0.1)
            results = np.loadtxt('{}{}_{}_Dmax=50_chi={}_Jij={}.txt'.format(
                result_dir, graph_pool[i], n[i], chi, Jij_pool[i]))
            exact = results[:, 1]
            tn = np.log10(abs(results[:, 2]).reshape(-1, 10) + 1e-20)
            results_van = np.loadtxt('{}{}_{}_Jij={}_van.txt'.format(
                result_dir, graph_pool[i], n[i], Jij_pool[i]))
            nmf = np.log10(abs(results[:, 4]).reshape(-1, 10))
            tap = np.log10(abs(results[:, 6]).reshape(-1, 10))
            bp = np.log10(abs(results[:, 8]).reshape(-1, 10))
            van = np.log10(abs(results_van[:, 1] - exact).reshape(-1, 10))
            ax[i].plot(beta, nmf.mean(axis=1), c='b', marker='x', label='NMF')
            ax[i].plot(beta, tap.mean(axis=1), c='g', marker="d", label='TAP')
            ax[i].plot(beta, bp.mean(axis=1), c='y', mfc='none', mec='y', marker="o", label='Bethe')
            ax[i].plot(beta, van.mean(axis=1), c='k', marker='*', label='VAN')
            ax[i].plot(beta, tn.mean(axis=1), c='r', marker='v', mfc='none', label='TN')
            '''
            Dmax_list = [1, 10, 20, 50]
            marker_list = ['<', '>', '^', 'v']
            for Dmax in Dmax_list:
                results = np.loadtxt('{}{}_{}_Dmax={}_chi={}_Jij={}.txt'.format(
                    result_dir, graph_pool[i], n[i], Dmax, chi, Jij_pool[i]))
                exact = results[:, 1]
                tn = np.log10(abs(results[:, 2]).reshape(-1, 10) + 1e-20)
                if Dmax == 1:
                    results_van = np.loadtxt('{}{}_{}_Jij={}_van.txt'.format(
                        result_dir, graph_pool[i], n[i], Jij_pool[i]))
                    nmf = np.log10(abs(results[:, 4]).reshape(-1, 10))
                    tap = np.log10(abs(results[:, 6]).reshape(-1, 10))
                    bp = np.log10(abs(results[:, 8]).reshape(-1, 10))
                    van = np.log10(abs(results_van[:, 1] - exact).reshape(-1, 10))
                    ax[i].plot(beta, nmf.mean(axis=1), c='b', marker='x', label='NMF')
                    ax[i].plot(beta, tap.mean(axis=1), c='tab:orange', marker="d", label='TAP')
                    ax[i].plot(beta, bp.mean(axis=1), c='y', mfc='none', mec='y', marker="o", label='Bethe')
                    ax[i].plot(beta, van.mean(axis=1), c='k', marker='*', label='VAN')
                ax[i].plot(beta, tn.mean(axis=1), c='r', marker=marker_list[Dmax_list.index(Dmax)], mfc='none',
                         label='Dmax: {}'.format(Dmax))
                """
                    plt.errorbar(beta, nmf.mean(axis=1), yerr=nmf.std(axis=1),
                                 c='b', fmt='-x', capsize=7, ms=3, linewidth=2, label='NMF')
                    plt.errorbar(beta, tap.mean(axis=1), yerr=tap.std(axis=1),
                                 c='tab:orange', fmt='-d', capsize=7, ms=3, linewidth=2, label='TAP')
                    plt.errorbar(beta, bp.mean(axis=1), yerr=bp.std(axis=1),
                                 c='y', fmt='-o', capsize=7, ms=3, linewidth=2, label='BP')
                    plt.errorbar(beta, van.mean(axis=1), yerr=van.std(axis=1),
                                 c='k', fmt='-*', capsize=7, ms=3, linewidth=2, label='VAN')
                plt.errorbar(beta, tn.mean(axis=1), yerr=tn.std(axis=1),
                             c='r', fmt='-{}'.format(marker_list[Dmax_list.index(Dmax)]),
                             capsize=7, ms=3, linewidth=2, label='Dmax: {}'.format(Dmax))
                """
            '''
            ax[i].set_ylim((-18, 0))
            ax[i].set_yticks([-1, -4, -7, -10, -13, -16])
            ax[i].set_yticklabels([], fontsize=40)
            # ax[i].set_yticklabels(['$10^{-1}$', '$10^{-4}$', '$10^{-7}$', '$10^{-10}$', '$10^{-13}$', '$10^{-16}$'],
            #                       fontsize=40)
            ax[i].set_xticks(beta[0:-1:4])
            # ax[i].set_xticklabels(ax[i].get_xticks(), fontsize=40)
            # ax[i].set_ylabel('Relative Error', fontsize=22)
            ax[i].set_xlabel(r'$\beta$', fontsize=45)
            ax[i].text(0.5, -0.2, annotate[i], fontdict=font_title, transform=ax[i].transAxes, ha="center")
            # ax[i].set_title(annotate[i], fontsize=50)
            graph_ax = ax[i].inset_axes([0.5, 0.275, 0.45, 0.45])
            graph_ax.set_axis_off()
            if i == 1:
                rrg = nx.random_regular_graph(3, D, seed=2)
                pos = nx.circular_layout(rrg)
                nodes = nx.draw_networkx_nodes(rrg, pos, ax=graph_ax)
                nodes.set_color('white')
                nodes.set_edgecolor('k')
                nx.draw_networkx_edges(rrg, pos, ax=graph_ax, width=3)
            elif i == 2:
                sw = nx.watts_strogatz_graph(D, 4, 0.4, seed=1)
                pos = nx.circular_layout(sw)
                nodes = nx.draw_networkx_nodes(sw, pos, ax=graph_ax)
                nodes.set_color('white')
                nodes.set_edgecolor('k')
                nx.draw_networkx_edges(sw, pos, ax=graph_ax, width=3)
            else:
                sk = nx.complete_graph(D)
                pos = nx.circular_layout(sk)
                nodes = nx.draw_networkx_nodes(sk, pos, ax=graph_ax)
                nodes.set_color('white')
                nodes.set_edgecolor('k')
                nx.draw_networkx_edges(sk, pos, ax=graph_ax)
                labels = ['Our method', 'Conv VAN', 'VAN', 'Bethe', 'TAP', 'NMF']
                labels.reverse()
                ax[i].legend(lines, labels,
                             loc='center left', bbox_to_anchor=(0, 0.44), ncol=1,
                             fontsize=40, frameon=False)
    plt.savefig('fig/relative_errorv2.eps', bbox_inches='tight', dpi=300)


def plot_combine_tn(result_dir, chi):
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rc('legend', fontsize=30)
    plt.rc('lines', lw=3, markersize=10, mew=2)
    plt.rc('axes', labelsize=30)
    fig, ax = plt.subplots(nrows=1, ncols=4)
    plt.subplots_adjust(wspace=0.05)
    ax = ax.flatten()
    # params['figure.figsize'] = [40, 10]
    # plt.rcParams.update(params)

    graph_pool = ['lattice', 'rrg', 'sw', 'complete']
    Jij_pool = ['ones', 'randn', 'randn', 'sk']
    n = [256, 80, 70, 20]
    for i in range(4):
        if i == 0:
            beta = np.arange(0.1, 1.1, 0.1)
            L = 16
            node = 'mps'
            if os.path.exists('results/squarelatticeL16/kacward{}.txt'.format(L)):
                exact = np.loadtxt('results/squarelatticeL16/kacward{}.txt'.format(L))

            # fe_tn = np.loadtxt('{}n256chi{}{}.txt'.format(result_dir, chi, node))
            fe_tn_np = np.loadtxt('results/squarelatticeL16/n256chi{}{}_np.txt'.format(chi, node))
            free_energy = np.loadtxt('results/squarelatticeL16/results.txt')
            fe_MF = np.loadtxt('results/squarelatticeL16/MFn{}.txt'.format(L ** 2))

            # left, bottom, width, height = 0.12, 0.12, 0.8, 0.8
            """
            ax1 = fig.add_axes([left, bottom, width, height])
            ax1.set_xlim((0.301, 1.02))
            ax1.set_ylim((-2.2, -1.85))
            ax1.set_yticks(np.arange(-2.2, -1.9, 0.1))
            ax1.tick_params(labelsize=15)
            ax1.set_xlabel(r'$\beta$', fontsize=22)
            ax1.set_ylabel('Free Energy', fontsize=22)
            ax1.plot(betas[33:], exact[33:], c='k')
            ax1.scatter(beta[3:], free_energy[1][3:], facecolors='none', edgecolors='y', marker='o')
            ax1.scatter(beta[3:], free_energy[3][3:], c='b', marker="x")
            ax1.scatter(beta[3:], free_energy[2][3:], c='tab:orange', marker="d")
            ax1.scatter(beta[3:], free_energy[0][3:], c='tab:cyan', marker='*')
            ax1.scatter(beta[3:], fe_tn[0][3:], c='r')
            ax1.legend(['Exact', 'Bethe', 'Dense', 'Conv', 'FVS', 'TN'], loc=2, ncol=1, fontsize=15, frameon=False)
            """
            # plt.axes([0.55, 0.2, 0.35, 0.45])
            # plt.axes([left, bottom, width, height])
            ax[0].axvline(0.4406868, color='k', linestyle='--', label='_nolegend_')
            """
            plt.scatter(beta, np.log10(np.abs(np.array(fe_tn[0]) - exact[0:100:11])),
                        c='r')
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[1]) - exact[0:100:11])),
                        facecolors='none', edgecolors='y', marker="o")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[2]) - exact[0:100:11])),
                        c='tab:orange', marker="d")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[3]) - exact[0:100:11])),
                        c='b', marker="x")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[0]) - exact[0:100:11])),
                        c='tab:cyan', marker='*')
            """
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[0]) - exact)), c='r', marker='<', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[2]) - exact)), c='r', marker='>', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[4]) - exact)), c='r', marker='^', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[6]) - exact)), c='r', marker='v', mfc='none')

            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[0]) - exact)),
                       c='y', marker='<', mfc='none', label='Dmax: 1')
            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[2]) - exact)),
                       c='b', marker='>', mfc='none', label='Dmax: 10')
            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[4]) - exact)),
                       c='tab:orange', marker='^', mfc='none', label='Dmax: 20')
            ax[0].plot(beta, np.log10(np.abs(np.array(fe_tn_np[6]) - exact)),
                       c='r', marker='v', mfc='none', label='Dmax: 50')
            '''
            ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[2]) - exact)),
                       c='c', marker="*", label='Conv')  # Conv
            ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[3]) - exact)),
                       c='k', marker='*', label='Dense')  # Dense
            # plt.plot(beta, np.log10(np.abs(np.array(free_energy[0]) - exact[0:100:11])),
            #          c='tab:cyan', marker='*')  # FVS
            ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[1]) - exact)),
                       c='y', mfc='none', mec='y', marker="o", label='Bethe')  # Bethe
            ax[0].plot(beta, np.log10(np.abs(fe_MF[1] - exact)),
                       c='tab:orange', marker="d", label='TAP')  # TAP
            ax[0].plot(beta, np.log10(np.abs(fe_MF[2] - exact)),
                       c='b', marker='x', label='NMF')  # NMF
            '''

            # ax[0].legend(loc=4, ncol=2, fontsize=15, frameon=False)
            ax[0].set_ylim((-18, 0))
            ax[0].set_yticks([-1, -4, -7, -10, -13, -16])
            ax[0].set_yticklabels(['$10^{-1}$', '$10^{-4}$', '$10^{-7}$', '$10^{-10}$', '$10^{-13}$', '$10^{-16}$'])
            ax[0].set_xticks(beta[0:10:2])
            ax[0].set_ylabel('Relative Error')
            ax[0].set_xlabel(r'$\beta$')
        else:
            beta = np.arange(0.1, 2.1, 0.1)
            Dmax_list = [1, 10, 20, 50]
            marker_list = ['<', '>', '^', 'v']
            color_list = ['y', 'b', 'tab:orange', 'r']
            for Dmax in Dmax_list:
                results = np.loadtxt('{}{}_{}_Dmax={}_chi={}_Jij={}.txt'.format(
                    result_dir, graph_pool[i], n[i], Dmax, chi, Jij_pool[i]))
                exact = results[:, 1]
                tn = np.log10(abs(results[:, 2]).reshape(-1, 10) + 1e-20)

                ax[i].plot(beta, tn.mean(axis=1), c=color_list[Dmax_list.index(Dmax)],
                           marker=marker_list[Dmax_list.index(Dmax)], mfc='none', label='Dmax: {}'.format(Dmax))
                """
                    plt.errorbar(beta, nmf.mean(axis=1), yerr=nmf.std(axis=1),
                                 c='b', fmt='-x', capsize=7, ms=3, linewidth=2, label='NMF')
                    plt.errorbar(beta, tap.mean(axis=1), yerr=tap.std(axis=1),
                                 c='tab:orange', fmt='-d', capsize=7, ms=3, linewidth=2, label='TAP')
                    plt.errorbar(beta, bp.mean(axis=1), yerr=bp.std(axis=1),
                                 c='y', fmt='-o', capsize=7, ms=3, linewidth=2, label='BP')
                    plt.errorbar(beta, van.mean(axis=1), yerr=van.std(axis=1),
                                 c='k', fmt='-*', capsize=7, ms=3, linewidth=2, label='VAN')
                plt.errorbar(beta, tn.mean(axis=1), yerr=tn.std(axis=1),
                             c='r', fmt='-{}'.format(marker_list[Dmax_list.index(Dmax)]),
                             capsize=7, ms=3, linewidth=2, label='Dmax: {}'.format(Dmax))
                """
            if i == 3:
                ax[i].legend(loc='center right', ncol=1, frameon=False)
            ax[i].set_ylim((-18, 0))
            ax[i].set_yticks([-1, -4, -7, -10, -13, -16])
            ax[i].set_yticklabels([])
            ax[i].set_xticks(beta[0:-1:4])
            # ax[i].set_ylabel('Relative Error', fontsize=22)
            ax[i].set_xlabel(r'$\beta$')
    plt.savefig('fig/relative_error_tnv1.eps', bbox_inches='tight', dpi=300)


def plot_combine_time(result_dir, chi):
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    plt.rc('legend', fontsize=30)
    plt.rc('lines', lw=3, markersize=10, mew=2)
    plt.rc('axes', labelsize=30)
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax = ax.flatten()

    graph_pool = ['lattice', 'rrg', 'sw', 'complete']
    Jij_pool = ['ones', 'randn', 'randn', 'sk']
    n = [256, 80, 70, 20]
    for i in range(4):
        if i == 0:
            beta = np.arange(0.1, 1.1, 0.1)
            L = 16
            node = 'mps'
            if os.path.exists('results/squarelatticeL16/kacward{}.txt'.format(L)):
                exact = np.loadtxt('results/squarelatticeL16/kacward{}.txt'.format(L))

            # fe_tn = np.loadtxt('{}n256chi{}{}.txt'.format(result_dir, chi, node))
            fe_tn_np = np.loadtxt('results/squarelatticeL16/n256chi{}{}_np.txt'.format(chi, node))
            free_energy = np.loadtxt('results/squarelatticeL16/results.txt')
            fe_MF = np.loadtxt('results/squarelatticeL16/MFn{}.txt'.format(L ** 2))

            # left, bottom, width, height = 0.12, 0.12, 0.8, 0.8
            """
            ax1 = fig.add_axes([left, bottom, width, height])
            ax1.set_xlim((0.301, 1.02))
            ax1.set_ylim((-2.2, -1.85))
            ax1.set_yticks(np.arange(-2.2, -1.9, 0.1))
            ax1.tick_params(labelsize=15)
            ax1.set_xlabel(r'$\beta$', fontsize=22)
            ax1.set_ylabel('Free Energy', fontsize=22)
            ax1.plot(betas[33:], exact[33:], c='k')
            ax1.scatter(beta[3:], free_energy[1][3:], facecolors='none', edgecolors='y', marker='o')
            ax1.scatter(beta[3:], free_energy[3][3:], c='b', marker="x")
            ax1.scatter(beta[3:], free_energy[2][3:], c='tab:orange', marker="d")
            ax1.scatter(beta[3:], free_energy[0][3:], c='tab:cyan', marker='*')
            ax1.scatter(beta[3:], fe_tn[0][3:], c='r')
            ax1.legend(['Exact', 'Bethe', 'Dense', 'Conv', 'FVS', 'TN'], loc=2, ncol=1, fontsize=15, frameon=False)
            """
            # plt.axes([0.55, 0.2, 0.35, 0.45])
            # plt.axes([left, bottom, width, height])
            ax[0].axvline(0.4406868, color='k', linestyle='--', label='_nolegend_')
            """
            plt.scatter(beta, np.log10(np.abs(np.array(fe_tn[0]) - exact[0:100:11])),
                        c='r')
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[1]) - exact[0:100:11])),
                        facecolors='none', edgecolors='y', marker="o")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[2]) - exact[0:100:11])),
                        c='tab:orange', marker="d")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[3]) - exact[0:100:11])),
                        c='b', marker="x")
            plt.scatter(beta, np.log10(np.abs(np.array(free_energy[0]) - exact[0:100:11])),
                        c='tab:cyan', marker='*')
            """
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[0]) - exact)), c='r', marker='<', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[2]) - exact)), c='r', marker='>', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[4]) - exact)), c='r', marker='^', mfc='none')
            # plt.plot(beta, np.log10(np.abs(np.array(fe_tn[6]) - exact)), c='r', marker='v', mfc='none')

            ax[0].plot(beta, np.array(fe_tn_np[1]),
                       c='y', marker='<', mfc='none', label='Dmax: 1')
            ax[0].plot(beta, np.array(fe_tn_np[3]),
                       c='b', marker='>', mfc='none', label='Dmax: 10')
            ax[0].plot(beta, np.array(fe_tn_np[5]),
                       c='tab:orange', marker='^', mfc='none', label='Dmax: 20')
            ax[0].plot(beta, np.array(fe_tn_np[7]),
                       c='r', marker='v', mfc='none', label='Dmax: 50')
            '''
            ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[2]) - exact)),
                       c='c', marker="*", label='Conv')  # Conv
            ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[3]) - exact)),
                       c='k', marker='*', label='Dense')  # Dense
            # plt.plot(beta, np.log10(np.abs(np.array(free_energy[0]) - exact[0:100:11])),
            #          c='tab:cyan', marker='*')  # FVS
            ax[0].plot(beta, np.log10(np.abs(np.array(free_energy[1]) - exact)),
                       c='y', mfc='none', mec='y', marker="o", label='Bethe')  # Bethe
            ax[0].plot(beta, np.log10(np.abs(fe_MF[1] - exact)),
                       c='tab:orange', marker="d", label='TAP')  # TAP
            ax[0].plot(beta, np.log10(np.abs(fe_MF[2] - exact)),
                       c='b', marker='x', label='NMF')  # NMF
            '''

            # ax[0].legend(loc=4, ncol=2, fontsize=15, frameon=False)
            ax[0].set_xticks(beta[0:10:2])
            ax[0].set_ylabel('Time(s)')
            ax[0].set_xlabel(r'$\beta$')
        else:
            beta = np.arange(0.1, 2.1, 0.1)
            Dmax_list = [1, 10, 20, 50]
            marker_list = ['<', '>', '^', 'v']
            color_list = ['y', 'b', 'tab:orange', 'r']
            for Dmax in Dmax_list:
                results = np.loadtxt('{}{}_{}_Dmax={}_chi={}_Jij={}.txt'.format(
                    result_dir, graph_pool[i], n[i], Dmax, chi, Jij_pool[i]))
                exact = results[:, 1]
                time_tn = results[:, 3].reshape(-1, 10)

                ax[i].plot(beta, time_tn.mean(axis=1), c=color_list[Dmax_list.index(Dmax)],
                           marker=marker_list[Dmax_list.index(Dmax)], mfc='none', label='Dmax: {}'.format(Dmax))
                """
                    plt.errorbar(beta, nmf.mean(axis=1), yerr=nmf.std(axis=1),
                                 c='b', fmt='-x', capsize=7, ms=3, linewidth=2, label='NMF')
                    plt.errorbar(beta, tap.mean(axis=1), yerr=tap.std(axis=1),
                                 c='tab:orange', fmt='-d', capsize=7, ms=3, linewidth=2, label='TAP')
                    plt.errorbar(beta, bp.mean(axis=1), yerr=bp.std(axis=1),
                                 c='y', fmt='-o', capsize=7, ms=3, linewidth=2, label='BP')
                    plt.errorbar(beta, van.mean(axis=1), yerr=van.std(axis=1),
                                 c='k', fmt='-*', capsize=7, ms=3, linewidth=2, label='VAN')
                plt.errorbar(beta, tn.mean(axis=1), yerr=tn.std(axis=1),
                             c='r', fmt='-{}'.format(marker_list[Dmax_list.index(Dmax)]),
                             capsize=7, ms=3, linewidth=2, label='Dmax: {}'.format(Dmax))
                """
            if i == 3:
                ax[i].legend(loc='center right', ncol=1, frameon=False)
            ax[i].set_xticks(beta[0:-1:4])
            # ax[i].set_ylabel('Relative Error', fontsize=22)
            ax[i].set_xlabel(r'$\beta$')
    plt.savefig('fig/time_tnv1.eps', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    plot_combine('results/', 500)
    plot_combine_tn('results/', 500)
    plot_combine_time('results/', 500)
