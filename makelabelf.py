import pandas as pd
from datetime import datetime, timedelta

class makehalfh:
    def __init__(self,start_date,end_date):
        self.start_date = start_date
        self.end_date = end_date


    # 创建时间范围
    def generate_timeline(self,start_date, end_date):
        timeline = pd.DataFrame(columns=['timeblock_start', 'timeblock_end', 'class'])
        current_date = start_date
        while current_date <= end_date:
            week_start = current_date - timedelta(days=current_date.weekday())
            week_end = week_start + timedelta(days=6)
            timeline = timeline._append({'timeblock_start': week_start, 'timeblock_end': week_end, 'class': 0},
                                       ignore_index=True)
            current_date += timedelta(hours=0.5)
        return timeline

    # 读取事件表并标记abnormaltime
    def mark_abnormaltime(self,timeline, event_data):
        for index, row in event_data.iterrows():
            event_time = row['Time']
            event_time = datetime.strptime(event_time, '%Y/%m/%d %H:%M')
            abnormal_start = event_time - timedelta(hours=4)
            abnormal_end = event_time + timedelta(hours=4)
            timeline.loc[(timeline['timeblock_start'] <= abnormal_end) & (
                        timeline['timeblock_end'] >= abnormal_start), 'class'] = 1 if \
                row['Type'] == 'Misconfiguration' else 2 if \
            row['Type'] == 'Hijack' else 3
        return timeline

    # 生成timeline


    def edit_time_range(self):
        timeline = self.generate_timeline(self.start_date, self.end_date)

        # 读取事件表
        event_data = pd.read_csv('bgp_event.csv')

        # 标记abnormaltime
        timeline = self.mark_abnormaltime(timeline, event_data)

        # 将结果写入csv文件
        timeline.to_csv('timeline_hh.csv', index=False)

class makeday:
    def __init__(self,start,end):
        # 创建时间线表
        self.start_date = start
        self.end_date = end

    def edit_time_range(self):
        date_range = pd.date_range(start=start_date, end=end_date)

        # 创建时间线表格
        timeline = pd.DataFrame(columns=['timeblock_start', 'timeblock_end', 'class'])
        timeline['timeblock_start'] = date_range
        timeline['timeblock_end'] = date_range + timedelta(days=6)  # 一周的结束时间是起始时间加6天
        timeline['class'] = 0  # 默认为0

        # 读取事件表
        events = pd.read_csv('bgp_event.csv')

        # 处理时间格式
        events['Time'] = pd.to_datetime(events['Time'])

        # 获取异常时间段
        events['abnomaltime_start'] = events['Time'] - timedelta(hours=4)
        events['abnomaltime_end'] = events['Time'] + timedelta(hours=4)

        # 标记异常时间段对应的class
        for index, event in events.iterrows():
            mask = (timeline['timeblock_start'] <= event['abnomaltime_end']) & \
                   (timeline['timeblock_end'] >= event['abnomaltime_start'])
            if event['Type'] == 'Misconfiguration':
                timeline.loc[mask, 'class'] = 1
            elif event['Type'] == 'Hijack':
                timeline.loc[mask, 'class'] = 2
            elif event['Type'] == 'leak':
                timeline.loc[mask, 'class'] = 3

        # 将结果写入CSV文件
        timeline.to_csv('timeline.csv', index=False)

class makeweek:
    def __init__(self,start_date,end_date):
        self.start_date = start_date
        self.end_date = end_date
    def edit_time_range(self):
        timeline = pd.DataFrame(columns=['timeblock_start', 'timeblock_end', 'class'])

        current_date = self.start_date
        while current_date <= self.end_date:
            # 获取一周的起止时间
            timeblock_start = current_date
            timeblock_end = current_date + timedelta(days=6)

            # 添加一行到时间表格，默认class为0
            timeline = timeline._append({'timeblock_start': timeblock_start,
                                        'timeblock_end': timeblock_end,
                                        'class': 0}, ignore_index=True)

            # 移动到下一周
            current_date += timedelta(weeks=1)

        # 读取事件表格
        events = pd.read_csv('bgp_event.csv')

        # 将事件时间转换为datetime对象
        events['Time'] = pd.to_datetime(events['Time'])

        # 定义abnormal时间范围
        abnormal_time_start = timedelta(days=-1)
        abnormal_time_end = timedelta(days=1)

        # 遍历事件，更新时间表格中的class
        for index, event in events.iterrows():
            event_time = event['Time']
            abnormal_start = event_time + abnormal_time_start
            abnormal_end = event_time + abnormal_time_end

            # 根据事件类型更新class
            event_type = event['Type']
            if event_type == 'Misconfiguration':
                timeline.loc[(timeline['timeblock_start'] <= abnormal_end) & (
                            timeline['timeblock_end'] >= abnormal_start), 'class'] = 1
            elif event_type == 'Hijack':
                timeline.loc[(timeline['timeblock_start'] <= abnormal_end) & (
                            timeline['timeblock_end'] >= abnormal_start), 'class'] = 2
            elif event_type == 'leak':
                timeline.loc[(timeline['timeblock_start'] <= abnormal_end) & (
                            timeline['timeblock_end'] >= abnormal_start), 'class'] = 3

        # 将结果写入CSV文件
        timeline.to_csv('timeline_week_events.csv', index=False)
#
# start_date = datetime(2010, 1, 1)
# end_date = datetime(2022, 12, 31)
# make_h = makehalfh(start_date,end_date)
# make_h.edit_time_range()
# start_date = datetime(2010, 1, 1)
# end_date = datetime(2022, 12, 31)
# make_d = makeday(start_date,end_date)
# make_d.edit_time_range()
start_date = datetime(2010,1,4)
end_date = datetime(2022,12,31)
make_w = makeweek(start_date,end_date)
make_w.edit_time_range()