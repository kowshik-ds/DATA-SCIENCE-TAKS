Original file is located at
    https://colab.research.google.com/drive/1X1Zwmp0iahdbtWmdC5B88RYvQ5O8uOuz

# Instagram Engagement Analysis for Alfido Tech
This notebook analyzes posting time, engagement per follower, content type, hashtags, and follower growth signals.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
comments = pd.read_csv("comments.csv")
follows = pd.read_csv("follows.csv")
likes = pd.read_csv("likes.csv")
photo_tags = pd.read_csv("/content/photo_tags.csv")
photos = pd.read_csv("photos.csv")
tags = pd.read_csv("tags.csv")
users = pd.read_csv("users.csv")

# Clean column names
for df in [comments, follows, likes, photo_tags, photos, tags, users]:
    df.columns = [c.strip().lower().replace(' ', '_').replace('__','_') for c in df.columns]

# Parse date/time columns
comments['created_dt'] = pd.to_datetime(comments['created_timestamp'], dayfirst=True, errors='coerce')
likes['created_dt'] = pd.to_datetime(likes['created_time'], dayfirst=True, errors='coerce')
follows['created_dt'] = pd.to_datetime(follows['created_time'], dayfirst=True, errors='coerce')
photos['created_dt'] = pd.to_datetime(photos['created_dat'], dayfirst=True, errors='coerce')
users['created_dt'] = pd.to_datetime(users['created_time'], dayfirst=True, errors='coerce')

# Follower count per creator account
follower_counts = follows.groupby('followee').agg(
    followers=('follower','nunique'),
    active_followers=('is_follower_active','sum')
).reset_index().rename(columns={'followee':'user_id'})

# Engagement per post
post_metrics = photos.rename(columns={'id':'photo_id', 'user_id':'creator_id'}).copy()
likes_count = likes.groupby('photo').size().rename('likes').reset_index().rename(columns={'photo':'photo_id'})
comments_count = comments.groupby('photo_id').size().rename('comments').reset_index()
hashtag_avg = comments.groupby('photo_id')['hashtags_used_count'].mean().rename('avg_hashtags_in_comments').reset_index()

# Hashtags for each photo
tag_join = photo_tags.rename(columns={'photo':'photo_id', 'tag_id':'id'}).merge(tags[['id','tag_text']], on='id', how='left')
post_tags = tag_join.groupby('photo_id').agg(
    tags=('tag_text', lambda x:', '.join(sorted(set(x.dropna().astype(str))))),
    tag_count=('tag_text','nunique')
).reset_index()

post_metrics = (post_metrics
    .merge(likes_count, on='photo_id', how='left')
    .merge(comments_count, on='photo_id', how='left')
    .merge(hashtag_avg, on='photo_id', how='left')
    .merge(post_tags, on='photo_id', how='left')
    .merge(follower_counts, left_on='creator_id', right_on='user_id', how='left'))

for c in ['likes','comments','followers','active_followers','tag_count']:
    post_metrics[c] = post_metrics[c].fillna(0)

post_metrics['total_engagement'] = post_metrics['likes'] + post_metrics['comments']
post_metrics['engagement_per_follower'] = np.where(post_metrics['followers'] > 0,
                                                   post_metrics['total_engagement'] / post_metrics['followers'],
                                                   np.nan)
post_metrics['likes_per_follower'] = np.where(post_metrics['followers'] > 0, post_metrics['likes'] / post_metrics['followers'], np.nan)
post_metrics['comments_per_follower'] = np.where(post_metrics['followers'] > 0, post_metrics['comments'] / post_metrics['followers'], np.nan)
post_metrics['hour'] = post_metrics['created_dt'].dt.hour
post_metrics['day_name'] = post_metrics['created_dt'].dt.day_name()
post_metrics['day_num'] = post_metrics['created_dt'].dt.dayofweek
post_metrics['photo_type'] = post_metrics['photo_type'].str.lower().str.strip()
post_metrics['insta_filter_used'] = post_metrics['insta_filter_used'].str.lower().str.strip()

post_metrics.head()

"""## Data limitation
All timestamp columns in this dataset contain the same date/time: **13-04-2023 08:04**. Because of that, the notebook can parse dates and show the schedule workflow, but it cannot honestly prove a real best posting hour/day from this file. The recommended calendar should be treated as a 4-week testing plan.

## Summary metrics
"""

print('Total posts:', len(post_metrics))
print('Total likes:', int(post_metrics['likes'].sum()))
print('Total comments:', int(post_metrics['comments'].sum()))
print('Average engagement per follower:', round(post_metrics['engagement_per_follower'].mean(), 3))

"""## Best posting schedule"""

best_slots = (post_metrics.groupby(['day_num','day_name','hour'])
              .agg(posts=('photo_id','count'), avg_engagement=('engagement_per_follower','mean'), avg_total=('total_engagement','mean'))
              .reset_index()
              .sort_values('avg_engagement', ascending=False))
best_slots.head(10)

ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
slot_pivot = post_metrics.pivot_table(values='engagement_per_follower', index='day_name', columns='hour', aggfunc='mean').reindex(ordered_days)
plt.figure(figsize=(9,4.8))
plt.imshow(slot_pivot, aspect='auto')
plt.xticks(range(len(slot_pivot.columns)), slot_pivot.columns, rotation=90)
plt.yticks(range(len(slot_pivot.index)), slot_pivot.index)
plt.colorbar(label='Engagement per follower')
plt.title('Posting Time Heatmap')
plt.tight_layout()
plt.show()

"""## Content type analysis"""

type_perf = post_metrics.groupby('photo_type').agg(posts=('photo_id','count'), avg_engagement=('engagement_per_follower','mean'), avg_likes=('likes','mean'), avg_comments=('comments','mean')).reset_index().sort_values('avg_engagement', ascending=False)
type_perf

plt.figure(figsize=(7,4))
plt.bar(type_perf['photo_type'], type_perf['avg_engagement'])
plt.title('Average Engagement per Follower by Content Type')
plt.xlabel('Content type')
plt.ylabel('Engagement per follower')
plt.tight_layout()
plt.show()

"""## Hashtag analysis"""

tag_perf = tag_join.merge(post_metrics[['photo_id','engagement_per_follower','total_engagement']], on='photo_id', how='left')
(tag_perf.groupby('tag_text')
 .agg(posts=('photo_id','nunique'), avg_engagement=('engagement_per_follower','mean'), avg_total=('total_engagement','mean'))
 .reset_index()
 .sort_values('avg_engagement', ascending=False)
 .head(10))

"""## Follower growth signals"""

follower_growth = follows.set_index('created_dt').resample('D').size().rename('new_follows').reset_index()
post_by_day = post_metrics.set_index('created_dt').resample('D').agg(posts=('photo_id','count'), engagement=('total_engagement','sum')).reset_index()
growth_signal = follower_growth.merge(post_by_day, on='created_dt', how='outer').fillna(0)
growth_signal.head()

plt.figure(figsize=(8,4))
plt.plot(growth_signal['created_dt'], growth_signal['new_follows'], marker='o')
plt.title('Follower Growth Signal: New Follows Over Time')
plt.xlabel('Date')
plt.ylabel('New follows')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""## Recommended content calendar
Use the top-performing time slots as primary publishing windows. For Alfido Tech, keep the calendar simple: educational carousel/photo posts, proof/result posts, project tips, and engagement prompts.
"""

recommended_calendar = best_slots.head(5)[['day_name','hour','avg_engagement','avg_total']]
recommended_calendar['recommended_content'] = ['Tech tip / carousel', 'Project showcase', 'Internship learning post', 'Poll/question post', 'Behind-the-scenes / team post'][:len(recommended_calendar)]
recommended_calendar

"""## 5 engagement strategies
1. Post in the top 3 engagement windows instead of random timing.
2. Use the highest-performing hashtag themes, but avoid hashtag stuffing.
3. Prioritize the content type with best engagement per follower.
4. Add a comment trigger in every caption, such as a question or mini task.
5. Track engagement per follower weekly; raw likes alone are misleading when follower count differs.
"""
