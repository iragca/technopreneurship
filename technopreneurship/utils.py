import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from technopreneurship.config import MPL_STYLE_DIR, FIGURES_DIR

plt.style.use(MPL_STYLE_DIR / "iragca_ml.mplstyle")


class Question:

    def __init__(self, data, question: str):
        self.data = data
        self.question = question
        self.questions = {
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

    def wordcloud(self):
        text = " ".join(self.data)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800, height=400, background_color="white", colormap="viridis"
        ).generate(text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Hide axes
        plt.title(f"Question {self.question} - {self.questions[self.question]}", fontsize=10)
        plt.savefig(
            FIGURES_DIR / f"{self.question}-wordcloud", bbox_inches="tight", transparent=True
        )

    def bag_of_words(self):

        vectorizer = CountVectorizer(stop_words="english")

        X = vectorizer.fit_transform(self.data)

        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    def barplot(self, top_n: int = 10):
        data = self.bag_of_words().sum().sort_values(ascending=False).head(top_n)
        if top_n == 10:
            plt.figure(figsize=(4, 2.5))
        else:
            plt.figure(figsize=(4, 2.5 + top_n * 0.1))

        sns.barplot(x=data.values, y=data.index, orient="h")
        plt.title(
            f"Question {self.question} - {self.questions[self.question]}\n(Top {top_n} words)",
            fontsize=10,
        )
        plt.xlabel("Count", color="gray")
        plt.ylabel("")
        plt.savefig(
            FIGURES_DIR / f"{self.question}-barplot", bbox_inches="tight", transparent=True
        )
