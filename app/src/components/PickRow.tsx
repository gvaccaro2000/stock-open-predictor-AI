import { Pressable, StyleSheet, Text, View } from 'react-native';
import { fmtPct } from '../api';
import { colors } from '../theme';
import type { Pick } from '../types';

export function PickRow({ pick, onPress }: { pick: Pick; onPress: () => void }) {
  const gap = pick.premarket_gap_pct;
  const gapColor = gap == null ? colors.muted : gap >= 0 ? colors.up : colors.down;
  const topReason = pick.reasons[0]?.text ?? 'No strong single signal';

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.row, pressed && { opacity: 0.6 }]}
      accessibilityRole="button"
      accessibilityLabel={`${pick.ticker}, rank ${pick.rank}, score ${pick.score}`}
    >
      <Text style={styles.rank}>{pick.rank}</Text>
      <View style={styles.middle}>
        <Text style={styles.ticker}>{pick.ticker}</Text>
        <Text style={styles.reason} numberOfLines={1}>{topReason}</Text>
      </View>
      <View style={styles.right}>
        <Text style={[styles.gap, { color: gapColor }]}>{fmtPct(gap)}</Text>
        <View style={styles.scoreBarBg}>
          <View style={[styles.scoreBar, { width: `${pick.score}%` }]} />
        </View>
        <Text style={styles.scoreLabel}>Score {pick.score}</Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: colors.card,
    borderRadius: 14, padding: 14, marginBottom: 10, borderWidth: 1, borderColor: colors.border,
  },
  rank: { color: colors.muted, fontSize: 15, width: 26, fontVariant: ['tabular-nums'] },
  middle: { flex: 1, marginRight: 10 },
  ticker: { color: colors.text, fontSize: 18, fontWeight: '700' },
  reason: { color: colors.muted, fontSize: 13, marginTop: 2 },
  right: { alignItems: 'flex-end', width: 90 },
  gap: { fontSize: 16, fontWeight: '600', fontVariant: ['tabular-nums'] },
  scoreBarBg: { height: 4, width: 70, backgroundColor: colors.border, borderRadius: 2, marginTop: 6 },
  scoreBar: { height: 4, backgroundColor: colors.accent, borderRadius: 2 },
  scoreLabel: { color: colors.muted, fontSize: 11, marginTop: 3 },
});
