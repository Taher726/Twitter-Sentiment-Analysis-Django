from django.shortcuts import render
from .forms import TweetForm
from .models import Tweet
from .functions import load_model, predict

def analyse_tweet(request):
    if request.method == "POST":
        form = TweetForm(request.POST)
        if form.is_valid():
            tweet_text = form.cleaned_data["tweet"]
            vectoriser, LRModel = load_model()
            result_df = predict(vectoriser, LRModel, [tweet_text])
            sentiment = result_df["sentiment"].iloc[0]

            """#Save the tweet and its sentiment in database
            Tweet.objects.create(text = tweet_text, sentiment = sentiment)"""

            return render(request, "result.html", {"sentiment":sentiment, "text":tweet_text})
    else:
        form=TweetForm()
    
    return render(request, "analyse_tweet.html", {"form":form})