from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import typer
from loguru import logger
from tqdm import tqdm
from wordcloud import WordCloud

from technopreneurship.config import (
    FIGURES_DIR,
    MPL_STYLE_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


def generate_plot(data: pl.Series, question, output_path: Path):

    questions = {
        "1": "How do you typically manage your time on a daily bases?",
        "2": "What does your daily routine look like?",
        "3": "Do you follow a set schedule for your everyday task?",
        "4": "How do you plan and schedule your tasks or events?",
        "5": "When your schedule becomes overwhelming, how do you prioritize tasks or events?",
        "6": "Do you use any apps or tools to help you manage your time effectively?",
        "7": "How do you handle situations where you need to adjust or sacrifice certain events or commitments?",
        "8": "Do you often find yourself forgetting important details, meetings, or events?",
        "9": "What strategies do you use to ensure you don’t forget critical information or appointments?",
        "10": "What alternative solutions or strategies do you use to improve your time management?",
    }

    text = " ".join(data)

    # Generate the word cloud
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Hide axes
    plt.title(f"Question {question}: {questions[str(question)]}")
    plt.savefig(output_path / f"question_{question}.png", bbox_inches="tight")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "Customer Discovery Group 4 - Sheet1.csv",
    output_path: Path = FIGURES_DIR,
    mpl_style: Path = MPL_STYLE_DIR / "iragca_ml.mplstyle",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plots from data...")

    logger.info(f"Using mplstyle: {mpl_style.name}")
    plt.style.use(MPL_STYLE_DIR / "iragca_ml.mplstyle")

    # Load data
    data = pl.read_csv(input_path)

    # Generate plots
    try:
        for question in tqdm(range(1, 11), unit="question"):
            question_data = data[str(question)]
            generate_plot(question_data, question, output_path)
    except Exception as e:
        logger.error(f"An error occurred plotting question {question}: {e}")

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
