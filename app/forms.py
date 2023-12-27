from django import forms

class TweetForm(forms.Form):
    tweet = forms.CharField(widget=forms.Textarea)