# Brian Lesko 
# 1/21/2024
# Data science studies, real time live data display

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time 
import customize_gui # streamlit GUI modifications
gui = customize_gui.gui()
import plot 
my_plot = plot.plotting()

def main():
    # Set up the app UI
    gui.clean_format(wide=True)
    Sidebar = gui.about(text = "This code implements a live data viewing feed")
    Title, subTitle, image_spot = st.empty(), st.sidebar.empty(), st.columns([1,5,1])[1].empty()
    with Title:
        st.title("Live Data Feed")

    # Initialize loop variables
    fig, ax = my_plot.get_colored_plt("#F6F6F3",'#335095','#D6D6D6')
    my_plot.set_adaptive_ax(ax)
    dynamic_plot, = ax.plot([], [], 'o', color='#335095', markersize=5)
    start_time = time.time()
    data_log = []
    times = []
    ax.set_ylim(-1.5, 1.5) # set the Y axis limits to be static

    while True:
        t = time.time() - start_time
        times.append(t)

        # simulate or retrieve data here 
        # noisy sinusoidal data
        data_log.append((np.array([t, np.sin(t)]) + np.random.randn(2) * 0.1)[1])

        # Update the axes limits
        ax.set_xlim(max(times)-10, max(times)+10)
        x_ticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{tick:.2f}' for tick in x_ticks])  # format tick labels

        # update the display
        dynamic_plot.set_data(times,data_log)
        #show the figure
        with image_spot:
            st.pyplot(fig) 

main()