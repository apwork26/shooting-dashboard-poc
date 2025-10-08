
"""
NRAI Shooting Analysis Dashboard - ENHANCED Analytics Engine v2.0
================================================================
Enhanced with additional metrics for comprehensive analysis:
- Running totals tracking
- Elimination progression analysis
- Country-wise success rates
- Tournament difficulty normalization
- Shot sequence visualization data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ShootingAnalyticsEngine:
    """Enhanced analytics engine for shooting performance analysis"""

    def __init__(self, qualification_df: pd.DataFrame, finals_df: pd.DataFrame):
        self.qual_df = qualification_df
        self.finals_df = finals_df

    # ============ QUALIFICATION ANALYSIS ============

    def calculate_qualification_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive qualification performance metrics"""
        results = []

        for _, athlete in self.qual_df.iterrows():
            # Extract series data
            series = [athlete[f'series_{i}'] for i in range(1, 7)]

            # Calculate consistency metrics
            consistency_score = np.std(series)
            avg_series = np.mean(series)
            best_series = max(series)
            worst_series = min(series)
            series_range = best_series - worst_series

            # Performance relative to field
            tournament_data = self.qual_df[self.qual_df['tournament'] == athlete['tournament']]

            # Calculate cutoff (8th place)
            eighth_place_scores = tournament_data[tournament_data['rank'] == 8]['total']
            if len(eighth_place_scores) > 0:
                eighth_place_score = eighth_place_scores.iloc[0]
            else:
                eighth_place_score = 632.0 if 'WC' in athlete['tournament'] else 630.0

            top_score = tournament_data['total'].max()
            distance_from_cutoff = athlete['total'] - eighth_place_score
            distance_from_top = top_score - athlete['total']

            # Calculate percentile rank
            tournament_scores = tournament_data['total'].values
            percentile_rank = (np.sum(tournament_scores <= athlete['total']) / len(tournament_scores)) * 100

            # Series pattern analysis
            series_trend = self._calculate_series_trend(series)
            weak_series = self._identify_weak_series(series, avg_series)

            results.append({
                'tournament': athlete['tournament'],
                'athlete_id': f"{athlete['name']}_{athlete['noc']}",
                'athlete_name': athlete['name'],
                'country': athlete['noc'],
                'gender': athlete['gender'],
                'rank': athlete['rank'],
                'bib_number': athlete['bib'],
                'total_score': athlete['total'],
                'qualified': athlete['qualified'],
                'series_avg': round(avg_series, 2),
                'consistency_score': round(consistency_score, 2),
                'best_series': best_series,
                'worst_series': worst_series,
                'series_range': round(series_range, 1),
                'distance_from_cutoff': round(distance_from_cutoff, 1),
                'distance_from_top': round(distance_from_top, 1),
                'percentile_rank': round(percentile_rank, 1),
                'safety_margin': round(abs(distance_from_cutoff), 1),
                'series_trend': series_trend,
                'weak_series_count': weak_series,
                'series_1': series[0],
                'series_2': series[1],
                'series_3': series[2],
                'series_4': series[3],
                'series_5': series[4],
                'series_6': series[5],
                'series_pattern': ' -> '.join([str(s) for s in series])
            })

        return pd.DataFrame(results)

    # ============ NEW: COUNTRY SUCCESS RATES ============

    def calculate_country_success_rates(self) -> pd.DataFrame:
        """Calculate country-wise qualification success rates"""
        qual_metrics = self.calculate_qualification_metrics()

        country_stats = []
        for country in qual_metrics['country'].unique():
            country_data = qual_metrics[qual_metrics['country'] == country]

            qualified_count = len(country_data[country_data['qualified']])
            total_athletes = len(country_data)
            success_rate = (qualified_count / total_athletes * 100) if total_athletes > 0 else 0

            country_stats.append({
                'country': country,
                'total_athletes': total_athletes,
                'qualified': qualified_count,
                'success_rate': round(success_rate, 1),
                'avg_score': round(country_data['total_score'].mean(), 1),
                'best_score': country_data['total_score'].max(),
                'worst_score': country_data['total_score'].min(),
                'avg_consistency': round(country_data['consistency_score'].mean(), 2),
                'tournaments': list(country_data['tournament'].unique())
            })

        return pd.DataFrame(country_stats).sort_values('success_rate', ascending=False)

    # ============ FINALS ANALYSIS ============

    def calculate_finals_metrics(self) -> pd.DataFrame:
        """Calculate comprehensive finals performance metrics"""
        results = []

        for _, athlete in self.finals_df.iterrows():
            shots = athlete.get('shots')
            if shots is None or shots == '':
                continue
            if isinstance(shots, (int, float)) and pd.isna(shots):
                continue
            # Convert shots from string to list if needed
            if isinstance(shots, str):
                try:
                    shots = [float(s.strip()) for s in shots.split(',') if s.strip()]
                except (ValueError, AttributeError):
                    continue
            elif isinstance(shots, (list, tuple)):
                # Already a list/tuple, ensure all are floats
                try:
                    shots = [float(s) for s in shots if s]
                except (ValueError, TypeError):
                    continue
                
            else:
                continue
            if not shots or len(shots) == 0:
                continue
            total_shots = len(shots)


            # Shot quality analysis
            gqs_count = len([s for s in shots if s >= 10.0])
            excellent_count = len([s for s in shots if s >= 10.5])
            poor_count = len([s for s in shots if s < 9.5])
            average_count = len([s for s in shots if 9.5 <= s < 10.0])

            # Percentages
            gqs_percentage = (gqs_count / total_shots) * 100
            excellent_percentage = (excellent_count / total_shots) * 100
            poor_percentage = (poor_count / total_shots) * 100
            average_percentage = (average_count / total_shots) * 100

            # Pressure performance analysis
            first_stage_shots, second_stage_shots, elimination_shots = self._split_finals_stages(shots)

            first_stage_avg = np.mean(first_stage_shots) if first_stage_shots else 0
            second_stage_avg = np.mean(second_stage_shots) if second_stage_shots else 0
            elimination_avg = np.mean(elimination_shots) if elimination_shots else 0

            pressure_performance = second_stage_avg - first_stage_avg

            # Final performance (last 5 shots)
            final_5_shots = shots[-5:] if len(shots) >= 5 else shots
            final_5_avg = np.mean(final_5_shots)

            # Recovery analysis
            recovery_analysis = self._analyze_recovery_patterns(shots)

            # Consistency metrics
            shot_consistency = np.std(shots)
            coefficient_of_variation = (shot_consistency / np.mean(shots)) * 100

            # Shot sequence analysis
            shot_trends = self._analyze_shot_trends(shots)

            # NEW: Running totals
            running_totals = self._calculate_running_totals(shots)

            results.append({
                'tournament': athlete['tournament'],
                'athlete_id': f"{athlete['name']}_{athlete['noc']}",
                'athlete_name': athlete['name'],
                'country': athlete['noc'],
                'gender': athlete['gender'],
                'rank': athlete['rank'],
                'bib_number': athlete['bib'],
                'total_score': athlete['total'],
                'num_shots': total_shots,
                'shot_avg': round(np.mean(shots), 2),

                # Shot quality metrics
                'gqs_count': gqs_count,
                'gqs_percentage': round(gqs_percentage, 1),
                'excellent_count': excellent_count,
                'excellent_percentage': round(excellent_percentage, 1),
                'poor_count': poor_count,
                'poor_percentage': round(poor_percentage, 1),
                'average_count': average_count,
                'average_percentage': round(average_percentage, 1),

                # Pressure performance
                'first_stage_avg': round(first_stage_avg, 2),
                'second_stage_avg': round(second_stage_avg, 2),
                'elimination_avg': round(elimination_avg, 2),
                'final_5_avg': round(final_5_avg, 2),
                'pressure_performance': round(pressure_performance, 2),

                # Consistency metrics
                'shot_consistency': round(shot_consistency, 2),
                'coefficient_of_variation': round(coefficient_of_variation, 1),

                # Recovery metrics
                'recovery_count': recovery_analysis['recovery_count'],
                'avg_recovery_time': recovery_analysis['avg_recovery_time'],
                'recovery_success_rate': recovery_analysis['success_rate'],
                'recovery_patterns': recovery_analysis['patterns'],

                # Shot trends
                'shot_trend': shot_trends['trend'],
                'trend_strength': shot_trends['strength'],

                # Raw data for visualization
                'shots_string': ','.join(map(str, shots)),
                'running_totals': ','.join(map(str, running_totals)),
                'first_10_shots': ','.join(map(str, shots[:10])),
                'last_10_shots': ','.join(map(str, shots[-10:])) if len(shots) >= 10 else ','.join(map(str, shots))
            })

        return pd.DataFrame(results)

    # ============ NEW: RUNNING TOTALS ============

    def _calculate_running_totals(self, shots: List[float]) -> List[float]:
        """Calculate running totals of shots"""
        if not shots:
            return []
    
        # Ensure all shots are floats (handle string data from CSV)
        try:
            shots_float = [float(s) for s in shots]
        except (ValueError, TypeError):
            return []
    
        # Calculate cumulative sum
        return [sum(shots_float[:i+1]) for i in range(len(shots_float))]

    # ============ NEW: ELIMINATION PROGRESSION ============

    def analyze_elimination_progression(self, athlete_name: str, tournament: str) -> Dict:
        """Analyze shot-by-shot progression through elimination rounds"""
        athlete_data = self.finals_df[
            (self.finals_df['name'] == athlete_name) & 
            (self.finals_df['tournament'] == tournament)
        ]

        if athlete_data.empty:
            return {}

        athlete = athlete_data.iloc[0]
        shots = athlete['shots']

        if len(shots) < 20:
            return {}

        # Standard finals format: 10 + 10 + elimination
        stages = {
            'first_comp_stage': {
                'shots': shots[:10],
                'avg': round(np.mean(shots[:10]), 2),
                'total': round(sum(shots[:10]), 1),
                'position': 'Top 8'
            },
            'second_comp_stage': {
                'shots': shots[10:20],
                'avg': round(np.mean(shots[10:20]), 2),
                'total': round(sum(shots[10:20]), 1),
                'position': 'Top 8'
            }
        }

        # Elimination rounds
        elimination_shots = shots[20:]
        if len(elimination_shots) >= 2:
            stages['elimination_8_to_6'] = {
                'shots': elimination_shots[:2],
                'avg': round(np.mean(elimination_shots[:2]), 2),
                'total': round(sum(elimination_shots[:2]), 1)
            }

        if len(elimination_shots) >= 4:
            stages['elimination_6_to_4'] = {
                'shots': elimination_shots[2:4],
                'avg': round(np.mean(elimination_shots[2:4]), 2),
                'total': round(sum(elimination_shots[2:4]), 1)
            }

        if len(elimination_shots) >= 6:
            stages['elimination_4_to_2'] = {
                'shots': elimination_shots[4:6],
                'avg': round(np.mean(elimination_shots[4:6]), 2),
                'total': round(sum(elimination_shots[4:6]), 1)
            }

        if len(elimination_shots) > 6:
            stages['final_medals'] = {
                'shots': elimination_shots[6:],
                'avg': round(np.mean(elimination_shots[6:]), 2),
                'total': round(sum(elimination_shots[6:]), 1),
                'position': f"Rank {athlete['rank']}"
            }

        return stages

    # ============ NEW: SHOT SEQUENCE FOR VISUALIZATION ============

    def generate_shot_sequence_data(self, athlete_name: str = None, tournament: str = None) -> pd.DataFrame:
        """Generate shot-by-shot data formatted for visualization"""
        sequence_data = []

        finals_subset = self.finals_df
        if athlete_name:
            finals_subset = finals_subset[finals_subset['name'] == athlete_name]
        if tournament:
            finals_subset = finals_subset[finals_subset['tournament'] == tournament]

        for _, athlete in finals_subset.iterrows():
            shots = athlete['shots']
            running_totals = self._calculate_running_totals(shots)

            for i, (shot, cumulative) in enumerate(zip(shots, running_totals), 1):
                # Determine stage
                if i <= 10:
                    stage = 'First Stage'
                elif i <= 20:
                    stage = 'Second Stage'
                else:
                    stage = 'Elimination'

                # Determine shot quality
                if shot >= 10.5:
                    quality = 'Excellent'
                elif shot >= 10.0:
                    quality = 'GQS'
                elif shot >= 9.5:
                    quality = 'Average'
                else:
                    quality = 'Poor'

                sequence_data.append({
                    'athlete_name': athlete['name'],
                    'tournament': athlete['tournament'],
                    'country': athlete['noc'],
                    'shot_number': i,
                    'shot_value': shot,
                    'running_total': round(cumulative, 1),
                    'stage': stage,
                    'quality': quality,
                    'final_rank': athlete['rank']
                })

        return pd.DataFrame(sequence_data)

    # ============ CROSS-TOURNAMENT ANALYSIS ============

    def cross_tournament_analysis(self) -> Dict:
        """Analyze athlete performance across multiple tournaments"""
        qual_metrics = self.calculate_qualification_metrics()

        athlete_tournaments = qual_metrics.groupby('athlete_id').agg({
            'tournament': lambda x: list(x),
            'rank': lambda x: list(x),
            'total_score': lambda x: list(x),
            'consistency_score': lambda x: list(x),
            'qualified': lambda x: list(x)
        }).reset_index()

        multi_tournament = athlete_tournaments[
            athlete_tournaments['tournament'].apply(len) > 1
        ].copy()

        analysis_results = {}

        for _, athlete in multi_tournament.iterrows():
            athlete_id = athlete['athlete_id']
            tournaments = athlete['tournament']
            ranks = athlete['rank']
            scores = athlete['total_score']
            consistency = athlete['consistency_score']

            score_trend = self._calculate_progression_trend(scores)
            rank_trend = self._calculate_progression_trend(ranks, reverse=True)

            analysis_results[athlete_id] = {
                'athlete_name': athlete_id.split('_')[0],
                'country': athlete_id.split('_')[1] if '_' in athlete_id else 'Unknown',
                'tournaments': tournaments,
                'tournament_count': len(tournaments),
                'scores': scores,
                'ranks': ranks,
                'consistency_scores': consistency,
                'score_progression': score_trend,
                'rank_progression': rank_trend,
                'best_performance': {
                    'tournament': tournaments[scores.index(max(scores))],
                    'score': max(scores),
                    'rank': ranks[scores.index(max(scores))]
                },
                'average_score': round(np.mean(scores), 1),
                'score_improvement': round(scores[-1] - scores[0], 1) if len(scores) >= 2 else 0,
                'rank_improvement': ranks[0] - ranks[-1] if len(ranks) >= 2 else 0
            }

        return analysis_results

    # ============ INSIGHTS GENERATION ============

    def generate_insights_report(self) -> Dict:
        """Generate comprehensive insights report"""
        qual_metrics = self.calculate_qualification_metrics()
        finals_metrics = self.calculate_finals_metrics()
        cross_tournament = self.cross_tournament_analysis()
        country_success = self.calculate_country_success_rates()

        indian_qual = qual_metrics[qual_metrics['country'] == 'IND']
        indian_finals = finals_metrics[finals_metrics['country'] == 'IND']

        insights = {
            'overall_statistics': {
                'total_athletes_analyzed': len(qual_metrics),
                'total_tournaments': len(qual_metrics['tournament'].unique()),
                'indian_athletes': len(indian_qual),
                'indian_qualification_rate': round(
                    len(indian_qual[indian_qual['qualified']]) / len(indian_qual) * 100, 1
                ) if len(indian_qual) > 0 else 0,
                'average_qualification_score': round(qual_metrics['total_score'].mean(), 1),
                'average_finals_score': round(finals_metrics['total_score'].mean(), 1) if not finals_metrics.empty else 0
            },

            'indian_performance': {
                'best_qualification_scores': indian_qual.nlargest(3, 'total_score')[
                    ['athlete_name', 'tournament', 'total_score', 'rank']
                ].to_dict('records') if not indian_qual.empty else [],

                'best_finals_performances': indian_finals.nlargest(3, 'total_score')[
                    ['athlete_name', 'tournament', 'total_score', 'rank']
                ].to_dict('records') if not indian_finals.empty else [],

                'most_consistent_qualifiers': indian_qual.nsmallest(3, 'consistency_score')[
                    ['athlete_name', 'tournament', 'consistency_score']
                ].to_dict('records') if not indian_qual.empty else [],

                'best_pressure_performers': indian_finals.nlargest(3, 'pressure_performance')[
                    ['athlete_name', 'tournament', 'pressure_performance']
                ].to_dict('records') if not indian_finals.empty else []
            },

            'cross_tournament_insights': {
                'multi_tournament_athletes': len(cross_tournament),
                'improving_athletes': [
                    athlete_id for athlete_id, data in cross_tournament.items()
                    if data['score_improvement'] > 0
                ],
                'most_improved': max(
                    cross_tournament.items(), 
                    key=lambda x: x[1]['score_improvement']
                ) if cross_tournament else None
            },

            'tournament_difficulty': self._analyze_tournament_difficulty(qual_metrics),
            'country_rankings': country_success.to_dict('records'),
            'recommendations': self._generate_recommendations(indian_qual, indian_finals, cross_tournament)
        }

        return insights

    # ============ HELPER METHODS ============

    def _calculate_series_trend(self, series: List[float]) -> str:
        first_half = np.mean(series[:3])
        second_half = np.mean(series[3:])

        if second_half > first_half + 0.5:
            return 'Improving'
        elif second_half < first_half - 0.5:
            return 'Declining'
        else:
            return 'Stable'

    def _identify_weak_series(self, series: List[float], avg: float) -> int:
        return len([s for s in series if s < avg - 0.5])

    def _split_finals_stages(self, shots: List[float]) -> Tuple[List[float], List[float], List[float]]:
        if len(shots) >= 20:
            first_stage = shots[:10]
            second_stage = shots[10:20]
            elimination = shots[20:]
        elif len(shots) >= 10:
            split = len(shots) // 2
            first_stage = shots[:split]
            second_stage = shots[split:]
            elimination = []
        else:
            first_stage = shots
            second_stage = []
            elimination = []

        return first_stage, second_stage, elimination

    def _analyze_recovery_patterns(self, shots: List[float]) -> Dict:
        poor_shots = [(i, shot) for i, shot in enumerate(shots) if shot < 9.5]
        recovery_times = []
        recovery_patterns = []

        for pos, poor_shot in poor_shots:
            if pos < len(shots) - 1:
                next_shots = shots[pos+1:pos+3]
                if next_shots and next_shots[0] >= 10.0:
                    recovery_times.append(1)
                    recovery_patterns.append(f"Shot {pos+1}: {poor_shot:.1f} -> {next_shots[0]:.1f}")
                elif len(next_shots) >= 2 and next_shots[1] >= 10.0:
                    recovery_times.append(2)
                    recovery_patterns.append(f"Shot {pos+1}: {poor_shot:.1f} -> {next_shots[0]:.1f} -> {next_shots[1]:.1f}")

        return {
            'recovery_count': len(recovery_times),
            'avg_recovery_time': round(np.mean(recovery_times), 1) if recovery_times else 0,
            'success_rate': len(recovery_times) / len(poor_shots) * 100 if poor_shots else 0,
            'patterns': '; '.join(recovery_patterns) if recovery_patterns else 'No poor shots'
        }

    def _analyze_shot_trends(self, shots: List[float]) -> Dict:
        if len(shots) < 5:
            return {'trend': 'Insufficient data', 'strength': 0}

        x = np.arange(len(shots))
        slope, _ = np.polyfit(x, shots, 1)

        if slope > 0.01:
            trend = 'Improving'
        elif slope < -0.01:
            trend = 'Declining'
        else:
            trend = 'Stable'

        strength = abs(slope) * 100
        return {'trend': trend, 'strength': round(strength, 2)}

    def _calculate_progression_trend(self, values: List[float], reverse: bool = False) -> str:
        if len(values) < 2:
            return 'Insufficient data'

        if reverse:
            if values[-1] < values[0]:
                return 'Improving'
            elif values[-1] > values[0]:
                return 'Declining'
        else:
            if values[-1] > values[0]:
                return 'Improving'
            elif values[-1] < values[0]:
                return 'Declining'

        return 'Stable'

    def _analyze_tournament_difficulty(self, qual_metrics: pd.DataFrame) -> Dict:
        tournament_stats = {}

        for tournament in qual_metrics['tournament'].unique():
            tournament_data = qual_metrics[qual_metrics['tournament'] == tournament]
            qualified_data = tournament_data[tournament_data['qualified']]

            tournament_stats[tournament] = {
                'avg_score': round(tournament_data['total_score'].mean(), 1),
                'score_std': round(tournament_data['total_score'].std(), 2),
                'qualification_cutoff': qualified_data['total_score'].min() if len(qualified_data) > 0 else 0,
                'athlete_count': len(tournament_data),
                'difficulty_index': round(qualified_data['total_score'].min() if len(qualified_data) > 0 else 630, 1)
            }

        return tournament_stats

    def _generate_recommendations(self, indian_qual: pd.DataFrame, 
                                   indian_finals: pd.DataFrame, 
                                   cross_tournament: Dict) -> List[str]:
        recommendations = []

        if not indian_qual.empty:
            avg_consistency = indian_qual['consistency_score'].mean()
            if avg_consistency > 1.0:
                recommendations.append("Focus on improving series consistency through training standardization")

            avg_distance_from_cutoff = indian_qual['distance_from_cutoff'].mean()
            if avg_distance_from_cutoff < 2.0:
                recommendations.append("Athletes qualifying with minimal safety margin - increase training intensity")

        if not indian_finals.empty:
            avg_gqs = indian_finals['gqs_percentage'].mean()
            if avg_gqs < 85:
                recommendations.append("Increase focus on shot quality to achieve >85% GQS rate")

            avg_pressure = indian_finals['pressure_performance'].mean()
            if avg_pressure < 0:
                recommendations.append("Implement pressure training - performance declining in second stage")

        improving_athletes = [data for data in cross_tournament.values() if data['score_improvement'] > 0]
        if len(improving_athletes) > 0:
            recommendations.append(f"Maintain training programs - {len(improving_athletes)} athletes showing improvement")

        return recommendations


if __name__ == "__main__":
    pass
