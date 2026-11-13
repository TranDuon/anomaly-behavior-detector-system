"""Briefly show a timestamped running event without extending model decisions."""


class RecentRunningAlert:
    def __init__(self, seconds=1.2):
        self.seconds = seconds
        self.confirmed_at = None

    def apply(self, result, timestamp):
        display = dict(result)
        if result.get('display_policy') != 'running_confirmed_geometry_v1':
            return display
        remembered = result.get('last_confirmed_running_at')
        if remembered is not None and 0 <= timestamp - remembered <= self.seconds:
            if self.confirmed_at is None or remembered > self.confirmed_at:
                self.confirmed_at = remembered
        if result.get('state') == 'fighting' and 'fast_running' in result.get('active_behaviors', []) and not result.get('score_stale'):
            observed = result.get('rgb_prediction_timestamp')
            if observed is None:
                observed = result.get('prediction_timestamp')
            if observed is not None and 0 <= timestamp - observed <= self.seconds:
                if self.confirmed_at is None or observed > self.confirmed_at:
                    self.confirmed_at = observed
        if (self.confirmed_at is not None and 0 <= timestamp - self.confirmed_at <= self.seconds
                and not result.get('active_behaviors', [])):
            display['recent_running_alert'] = {'confirmed_at': self.confirmed_at}
        return display
