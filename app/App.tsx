import { StatusBar } from 'expo-status-bar';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, FlatList, RefreshControl, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { fetchReport } from './src/api';
import { PickDetail } from './src/components/PickDetail';
import { PickRow } from './src/components/PickRow';
import { colors } from './src/theme';
import type { Pick, Report } from './src/types';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <TodayScreen />
    </SafeAreaProvider>
  );
}

function TodayScreen() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selected, setSelected] = useState<Pick | null>(null);

  const load = useCallback(async () => {
    try {
      setError(null);
      setReport(await fetchReport());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load data');
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const onRefresh = async () => {
    setRefreshing(true);
    await load();
    setRefreshing(false);
  };

  if (!report && !error) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator color={colors.accent} />
      </View>
    );
  }

  const updated = report
    ? new Date(report.generated_at).toLocaleString(undefined, {
        weekday: 'short', hour: '2-digit', minute: '2-digit',
      })
    : '';

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <FlatList
        data={report?.picks ?? []}
        keyExtractor={(p) => p.ticker}
        contentContainerStyle={{ padding: 16, paddingBottom: 40 }}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.muted} />}
        ListHeaderComponent={
          <View style={{ marginBottom: 16 }}>
            <Text style={styles.title}>Open Signals</Text>
            {report && (
              <Text style={styles.subtitle}>
                {report.trading_date} · updated {updated} · {report.universe_size} stocks scanned
              </Text>
            )}
            {error && <Text style={styles.error}>{error}. Pull down to retry.</Text>}
            {report && bannerText(report) && (
              <View style={styles.banner}>
                <Text style={styles.bannerText}>{bannerText(report)}</Text>
              </View>
            )}
          </View>
        }
        renderItem={({ item }) => <PickRow pick={item} onPress={() => setSelected(item)} />}
        ListFooterComponent={
          report ? <Text style={styles.disclaimer}>{report.disclaimer}</Text> : undefined
        }
      />
      <PickDetail pick={selected} disclaimer={report?.disclaimer ?? ''} onClose={() => setSelected(null)} />
    </SafeAreaView>
  );
}

function bannerText(r: Report): string | null {
  if (r.model === 'demo') return 'Demo data — real picks appear after the first morning run.';
  if (r.run_window && r.run_window !== 'premarket')
    return 'Test run outside premarket hours — signals are not meaningful. The real run is at 8:30 New York time on trading days.';
  if (!r.has_premarket_data) return 'No premarket data yet — scores use previous-day signals only.';
  return null;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: { justifyContent: 'center', alignItems: 'center' },
  title: { color: colors.text, fontSize: 30, fontWeight: '800' },
  subtitle: { color: colors.muted, fontSize: 13, marginTop: 4 },
  error: { color: colors.down, marginTop: 10 },
  banner: { backgroundColor: colors.warnBg, borderRadius: 10, padding: 10, marginTop: 12 },
  bannerText: { color: colors.warnText, fontSize: 13 },
  disclaimer: { color: colors.muted, fontSize: 12, marginTop: 16, lineHeight: 17 },
});
