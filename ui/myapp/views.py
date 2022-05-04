# Create your views here.

import json
import traceback
import sys
import csv
import os
import pandas as pd

from functools import reduce
from operator import and_

from django.shortcuts import render
from django import forms

from analyze_reddit import go


class SearchForm(forms.Form):
    groups = forms.CharField(
        label='Subreddits to compare',
        help_text='Please separate each Reddit group by a space '
        '(e.g. trump JoeBiden)',
        required=True)
    start_date = forms.CharField(
        label="Start date",
        help_text='Format: yyyy-mm-dd (default=2022-3-1)',
        required=False)
    end_date = forms.CharField(
        label="End date",
        help_text='Format: yyyy-mm-dd (default=2022-4-1)',
        required=False)
    num = forms.IntegerField(
        label='# of posts to include in each subreddit',
        help_text='Please enter an integer (default=500)',
        required=False)
    show_corr = forms.BooleanField(label='Show statistical correlations',
                                   required=False)
    show_hotwords = forms.BooleanField(label='Show hot words',
                                   required=False)
    show_frequser = forms.BooleanField(label='Show frequent users',
                                   required=False)
    ppw = forms.IntegerField(
        label='Posts per week',
        help_text='Shreshold for a user to be considered "frequent poster"'
        ' (default=10)',
        required=False)


def index(request):
    context = {}
    # default inputs
    res = None
    show_corr = False
    show_frequser = False
    show_hotwords = False
    start_date = [2022, 3, 1]
    end_date = [2022, 4, 1]
    num = 500
    ppw = 10

    if request.method == 'GET':
        # create a form instance and populate it with data from the request:
        form = SearchForm(request.GET)
            # check whether it's valid:
        if form.is_valid():
            all_groups = form.cleaned_data['groups']
            group_lst = all_groups.split()
            print(form.cleaned_data['start_date'])
            if form.cleaned_data['start_date'] != '':
                start_date = form.cleaned_data['start_date'].split("-")
                start_date = [int(each) for each in start_date]
            if form.cleaned_data['end_date'] != '':
                end_date = form.cleaned_data['end_date'].split("-")
                end_date = [int(each) for each in end_date]
            if form.cleaned_data['num']:
                num = form.cleaned_data['num']
            if form.cleaned_data['show_corr']:
                show_corr = True
            if form.cleaned_data['show_frequser']:
                show_frequser = True
            if form.cleaned_data['show_hotwords']:
                show_hotwords = True
            if form.cleaned_data['ppw']:
                ppw = form.cleaned_data['ppw']
            title = all_groups + " sentiment comparison"

            try:
                if show_frequser or show_hotwords or show_corr:
                    res = go(group_lst, start_date, end_date,
                    num, ppw, title, show_frequser, show_hotwords, show_corr)
                else:
                    go(group_lst, start_date, end_date,
                    num, ppw, title, show_frequser, show_hotwords, show_corr)
                context['graph_img'] = title + '.png'
            except Exception as e:
                print('Exception caught')
    else:
        form  = SearchForm()


        # Convert form data to an args dictionary for find_courses

    context['form'] = form
    if res != None:
        if show_frequser or show_hotwords:
            context['group_lst'] = " and ".join(group_lst)
            if show_frequser:
                frequser = res[0]
                header, data = frequser[0], frequser[1]
                context['header'] = header
                context['res'] = data
            if show_hotwords:
                context['wordclouds'] = res[1]
        if show_corr:
            context['corr'] = res[2]



    return render(request, 'index.html', context)
