import pandas as pd
import numpy as np

class OilFieldFeatureEngineer:
    """Класс для инженерии признаков для датасета нефтяных месторождений"""

    def __init__(self, df):
        self.df = df.copy()

    # ============ 1. ГЕОГРАФИЧЕСКИЕ ФИЧИ ============
    def create_geographic_features(self):
        """Создание географических признаков"""
        df = self.df

        # Географический кластер 4x4
        lat_cluster = pd.cut(df['Latitude'].fillna(90), bins=10, labels=False)
        lon_cluster = pd.cut(df['Longitude'].fillna(0), bins=20, labels=False)
        df['geo_cluster'] = lat_cluster.astype(str) + '_' + lon_cluster.astype(str)

        return df

    # ============ 4. РЕЗЕРВУАРНЫЕ ФИЧИ ============
    def create_reservoir_features(self):
        """Создание резервуарных признаков"""
        df = self.df

        # Коэффициент Net-to-Gross
        df['net_to_gross_ratio'] = df['Thickness (net pay average ft)'] / (df['Thickness (gross average ft)'] + 1)

        # Качество пористости
        def porosity_quality(p):
            if p < 10:
                return 'Poor'
            elif p < 15:
                return 'Fair'
            elif p < 25:
                return 'Good'
            else:
                return 'Excellent'

        df['porosity_quality'] = df['Porosity'].apply(porosity_quality)

        # Flow Potential Index (нормализованный)
        df['flow_potential_index'] = (df['Permeability'] * df['Porosity']) / (df['Depth'] / 1000 + 1)

        # Reservoir Productivity
        df['reservoir_productivity'] = (df['Permeability'] * df['Thickness (net pay average ft)']) / (df['Depth'] / 1000 + 1)

        # Толщинные соотношения
        df['thickness_ratio'] = df['Thickness (gross average ft)'] / (df['Thickness (net pay average ft)'] + 1)
        df['thickness_to_depth'] = df['Thickness (net pay average ft)'] / (df['Depth'] + 1)

        # Глубинные зоны
        df['depth_zone'] = pd.cut(df['Depth'], 
                                   bins=[0, 3000, 7000, 12000, 25000],
                                   labels=['Shallow', 'Moderate', 'Deep', 'Very_Deep'])

        # Квантильный ранг глубины
        df['depth_quartile'] = pd.qcut(df['Depth'], q=4, labels=['Q1_Shallow', 'Q2', 'Q3', 'Q4_Deep'], duplicates='drop')

        return df

    # ============ 7. СТАТУС МЕСТОРОЖДЕНИЯ ============
    def create_status_features(self):
        """Создание признаков статуса месторождения"""
        df = self.df

        # Зрелость месторождения
        def get_maturity(status):
            s = str(status).upper()
            if 'UNDER' in s or 'EARLY' in s:
                return 'Young'
            elif 'PRODUCTION' in s or 'REJUVENATING' in s:
                return 'Active'
            elif 'DECLINING' in s:
                return 'Declining'
            elif 'DEPLETED' in s:
                return 'Depleting'
            else:
                return 'Unknown'

        df['status_maturity'] = df['Reservoir status'].apply(get_maturity)

        # Производит ли месторождение
        df['is_producing'] = (~df['Reservoir status'].str.contains('UNKNOWN|DEPLETED', case=False, na=False)).astype(int)
        df['is_rejuvenating'] = df['Reservoir status'].str.contains('REJUVENATING', case=False, na=False).astype(int)

        return df

    # ============ 8. КОМПАНЬЙСКИЕ ФИЧИ ============
    def create_company_features(self):
        """Создание компаньйских признаков"""
        df = self.df

        # Размер компании по количеству месторождений
        company_counts = df['Operator company'].value_counts()
        df['company_field_count'] = df['Operator company'].map(company_counts)

        def categorize_company_size(count):
            if count > 5:
                return 'Major'
            elif count >= 2:
                return 'Medium'
            else:
                return 'Minor'

        df['company_size'] = df['company_field_count'].apply(categorize_company_size)

        # Опыт компании с глубокими месторождениями
        company_deep = df.groupby('Operator company')['Depth'].apply(
            lambda x: (x > 7000).sum() / len(x)
        )
        df['company_deep_ratio'] = df['Operator company'].map(company_deep)

        return df

    # ============ 9. БАССЕЙНОВЫЕ ФИЧИ ============
    def create_basin_features(self):
        """Создание признаков бассейна"""
        df = self.df

        # Количество месторождений в бассейне
        basin_counts = df['Basin name'].value_counts()
        df['basin_field_count'] = df['Basin name'].map(basin_counts)

        # Средняя глубина в бассейне
        basin_depth_mean = df.groupby('Basin name')['Depth'].mean()
        df['basin_avg_depth'] = df['Basin name'].map(basin_depth_mean)

        # Средняя производительность в бассейне
        basin_productivity = df.groupby('Basin name').apply(
            lambda x: ((x['Permeability'] * x['Thickness (net pay average ft)']) / (x['Depth'] + 1)).mean()
        )
        df['basin_avg_productivity'] = df['Basin name'].map(basin_productivity)

        # Разница между месторождением и бассейном
        df['depth_vs_basin'] = df['Depth'] - df['basin_avg_depth']

        return df

    # ============ 10. ИНТЕГРАЦИЯ ============
    def create_integrated_features(self):
        """Создание интегрированных признаков"""
        df = self.df

        # Экономическая жизнеспособность (0-5)
        productivity_norm = (df['reservoir_productivity'] - df['reservoir_productivity'].min()) / (df['reservoir_productivity'].max() - df['reservoir_productivity'].min() + 1) * 2
        status_score = df['status_maturity'].map({'Young': 1, 'Active': 2, 'Declining': 1.5, 'Depleting': 0.5, 'Unknown': 1}).fillna(1) / 2
        df['economic_viability'] = (productivity_norm + status_score + df['basin_field_count']/20) / 2
        df['economic_viability'] = df['economic_viability'].clip(0, 5).round(2)

        return df

    # ============ 11. НОРМАЛИЗАЦИЯ ============
    def normalize_numeric_features(self):
        """Нормализация числовых признаков"""
        df = self.df

        # Log-трансформация для признаков со степенными законами
        df['log_depth'] = np.log1p(df['Depth'])
        df['log_permeability'] = np.log1p(df['Permeability'])
        df['log_porosity'] = np.log1p(df['Porosity'])
        df['log_thickness'] = np.log1p(df['Thickness (net pay average ft)'])

        return df

    # ============ ГЛАВНЫЙ МЕТОД ============
    def engineer_all_features(self):
        """Применить все трансформации признаков"""
        self.df = self.create_geographic_features()
        self.df = self.create_reservoir_features()
        self.df = self.create_status_features()
        self.df = self.create_company_features()
        self.df = self.create_basin_features()
        self.df = self.create_integrated_features()
        self.df = self.normalize_numeric_features()
        return self.df

# ============ ИСПОЛЬЗОВАНИЕ ============
if __name__ == "__main__":
    # Загрузить данные
    df = pd.read_csv('train_oil.csv')

    # Применить инженерию признаков
    engineer = OilFieldFeatureEngineer(df)
    df_engineered = engineer.engineer_all_features()

    # Информация о новых признаках
    original_cols = set(df.columns)
    new_cols = set(df_engineered.columns) - original_cols

    print(f"\nИсходное количество признаков: {len(original_cols)}")
    print(f"Новых признаков создано: {len(new_cols)}")
    print(f"Итого признаков: {len(df_engineered.columns)}")

    print(f"\nНовые признаки:({len(sorted(new_cols))})")
    for i, col in enumerate(sorted(new_cols), 1):
        print(f"  {i}. {col}")

    # Сохранить результат
    df_engineered.to_csv('train_oil_engineered.csv', index=False)
    print("\n✓ Сохранено в train_oil_engineered.csv")
