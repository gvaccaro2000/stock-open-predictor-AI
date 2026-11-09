import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { fmtPct, fmtPrice } from '../api';
import { colors } from '../theme';
import type { Pick } from '../types';

type Props = { pick: Pick | null; disclaimer: string; onClose: () => void };

export function PickDetail({ pick, disclaimer, onClose }: Props) {
  return (
    <Modal visible={pick != null} animationType="slide" presentationStyle="pageSheet"
           onRequestClose={onClose}>
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        {pick && (
          <ScrollView contentContainerStyle={{ padding: 20 }}>
            <View style={styles.header}>
              <Text style={styles.ticker}>{pick.ticker}</Text>
              <Pressable onPress={onClose} hitSlop={12} accessibilityRole="button">
                <Text style={styles.close}>Close</Text>
              </Pressable>
            </View>
            <Text style={styles.sub}>Rank #{pick.rank} · Score {pick.score}/100</Text>

            <Text style={styles.section}>Why it's on the list</Text>
            {pick.reasons.length === 0 && <Text style={styles.muted}>No single dominant signal.</Text>}
            {pick.reasons.map((r, i) => (
              <View key={i} style={styles.reason}>
                <Text style={{ color: r.impact === 'positive' ? colors.up : colors.down, fontSize: 16 }}>
                  {r.impact === 'positive' ? '▲' : '▼'}
                </Text>
                <Text style={styles.reasonText}>{r.text}</Text>
              </View>
            ))}

            <Text style={styles.section}>Signals</Text>
            <Stat label="Previous close" value={fmtPrice(pick.prev_close)} />
            <Stat label="Premarket price" value={fmtPrice(pick.premarket_price)} />
            <Stat label="Premarket gap" value={fmtPct(pick.premarket_gap_pct)} />
            <Stat label="5-day momentum" value={fmtPct(pick.momentum_5d_pct)} />
            <Stat label="vs 50-day average" value={fmtPct(pick.trend_vs_ma50_pct)} />
            <Stat label="RSI (14)" value={pick.rsi_14?.toFixed(0) ?? '—'} />
            <Stat label="Volume vs normal"
                  value={pick.rel_volume_1d == null ? '—' : `${pick.rel_volume_1d.toFixed(1)}x`} />

            <Text style={styles.disclaimer}>{disclaimer}</Text>
          </ScrollView>
        )}
      </SafeAreaView>
    </Modal>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.muted}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  ticker: { color: colors.text, fontSize: 32, fontWeight: '800' },
  close: { color: colors.accent, fontSize: 17 },
  sub: { color: colors.muted, fontSize: 15, marginTop: 4 },
  section: { color: colors.text, fontSize: 17, fontWeight: '700', marginTop: 28, marginBottom: 10 },
  reason: { flexDirection: 'row', gap: 10, marginBottom: 10, alignItems: 'flex-start' },
  reasonText: { color: colors.text, fontSize: 15, flex: 1, lineHeight: 21 },
  stat: {
    flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 11,
    borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border,
  },
  statValue: { color: colors.text, fontSize: 15, fontVariant: ['tabular-nums'] },
  muted: { color: colors.muted, fontSize: 15 },
  disclaimer: { color: colors.muted, fontSize: 12, marginTop: 30, lineHeight: 17 },
});
