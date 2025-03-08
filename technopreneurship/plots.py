from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import typer
from loguru import logger
from tqdm import tqdm


from technopreneurship.config import (
    FIGURES_DIR,
    MPL_STYLE_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

from technopreneurship.utils import Question

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "Customer Discovery Group 4 - Sheet1.csv",
    output_path: Path = FIGURES_DIR,
    mpl_style: Path = MPL_STYLE_DIR / "iragca_ml.mplstyle",
    top_n: int = 10,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info(f"Using mplstyle: {mpl_style.name}")
    plt.style.use(MPL_STYLE_DIR / "iragca_ml.mplstyle")

    # Load data
    data = pl.read_csv(input_path)

    # Generate plots
    try:
        logger.info(f"Generating plots from data: {top_n=}")
        for question in tqdm(range(1, 11), unit="question", ncols=72):
            question_data = data[str(question)]
            Question(question_data, str(question)).wordcloud()
            Question(question_data, str(question)).barplot(top_n=top_n)

    except Exception as e:
        logger.error(f"An error occurred plotting question {question}: {e}")

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
