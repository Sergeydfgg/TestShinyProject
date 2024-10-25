from shiny import ui, render, App
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, expon, norm, poisson
import yfinance as yf
import pandas as pd

app_ui = ui.page_fluid(
    ui.h4("Непрерывные распределения"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "cont_dist_type",
                "Тип распределения:",
                {"expon": "Экспоненциальное", "norm": "Нормальное"}
            ),
            ui.input_slider("n_cont", "Размер выборки:", 10, 1000, 10)
        ),
        ui.output_plot("cont_dist_plot")
    ),
    ui.h4("Дискретные распределения"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "disc_dist_type",
                "Тип распределения:",
                {"binom": "Биноминальное", "poisson": "Пуассона"}
            ),
            ui.input_slider("n_disc", "Размер выборки:", 1, 100, 2)
        ),
        ui.output_plot("disc_dist_plot")
    ),
    ui.h4("Данные из Yahoo Finance"),
    ui.output_plot("yahoo_plot"),
)


def server(input, output, session):
    @output
    @render.plot
    def cont_dist_plot():
        n = input.n_cont()
        dist_type = input.cont_dist_type()

        fig, ax = plt.subplots()

        if dist_type == "expon":
            scale = 1
            x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), n)
            y = expon.pdf(x, scale=scale)
            ax.bar(x, y)
            ax.set_title(f"Экспоненциальное распределение (scale={scale})")

        elif dist_type == "norm":
            mu = 0
            sigma = 1
            x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), n)
            y = norm.pdf(x, mu, sigma)
            ax.bar(x, y)
            ax.set_title(f"Нормальное распределение (mu={mu}, sigma={sigma})")

        return fig

    @output
    @render.plot
    def disc_dist_plot():
        n = input.n_disc()
        dist_type = input.disc_dist_type()

        fig, ax = plt.subplots()

        if dist_type == "binom":
            p = 0.5
            x = np.arange(0, n + 1)
            y = binom.pmf(x, n, p)
            ax.bar(x, y)
            ax.set_title(f"Биноминальное распределение (n={n}, p={p})")

        elif dist_type == "poisson":
            mu = n / 2
            x = np.arange(0, n + 1)
            y = poisson.pmf(x, mu)
            ax.bar(x, y)
            ax.set_title(f"Распределение Пуассона (lambda={mu:.1f})")

        return fig

    @output
    @render.plot
    def yahoo_plot():
        data = yf.download("AAPL", period="1y")
        closing_prices = data["Close"]

        fig, ax = plt.subplots()
        ax.hist(closing_prices, bins=30, density=True, alpha=0.6, color='skyblue', label="Цена закрытия")

        mu, sigma = norm.fit(closing_prices)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        ax.plot(x, p, 'k', linewidth=2, label=f"Нормальное распределение (mu={mu:.2f}, sigma={sigma:.2f})")

        ax.set_title("Гистограмма цен закрытия акций Apple и нормальное распределение")
        ax.set_xlabel("Цена закрытия")
        ax.set_ylabel("Плотность")
        ax.legend()

        return fig


app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
