from datetime import datetime, timedelta
from statistics import mean, median, stdev


def parsar_time(ts: int):
    if len(str(ts)) > 10:
        ts = int(str(ts)[:10])
    ts = datetime.fromtimestamp(ts)
    ts_prev = ts - timedelta(hours=3)
    cur = datetime.strftime(ts - timedelta(hours=0), '%Y-%m-%dT%H:%M:%S.%fZ')
    prev = datetime.strftime(ts_prev, '%Y-%m-%dT%H:%M:%S.%fZ')
    return prev, cur


def calculate_statical(timeseries) -> dict:
    values = [item[0] for item in timeseries]
    _mean = mean(values)
    _median = median(values)
    _stdev = stdev(values)
    _min = min(values)
    _max = max(values)
    out = []
    for item in timeseries:
        val, ts = item
        if val > _mean + 3 * _stdev:
            out.append(ts)
    return {
        'mean': _mean,
        'median': _median,
        'max': _max,
        'min': _min,
        'stdev': _stdev,
        'exceeding_3_sigma_timestamp': out
    }

