//! Numeric dates of the interface, year first: `2026/10/15`, and
//! `10/15` when the year is not shown. The one place that decides how a
//! numeric date reads, in every language. Dates with a month name stay in
//! the catalogues (`date-day-month-year`, `ui-chat-day`…), which put the
//! year first too; the catalogue keys that take numeric parts
//! (`profile-date`, `inspect-date`…) read like these helpers in English and
//! French, which `tests/catalogs.rs` checks, and keep their 年月日 wording
//! in Japanese. Dates sent to a server, written in a file name or a log
//! line are not interface text and do not come here.
// aurora-local-patch: year-first-dates-helper

/// `2026/10/15`: the year on four digits, the month and the day on two.
pub fn numeric_date(year: i64, month: i64, day: i64) -> String {
    format!("{year:04}/{month:02}/{day:02}")
}

/// `10/15`: a numeric date shown without its year, month then day.
pub fn numeric_month_day(month: i64, day: i64) -> String {
    format!("{month:02}/{day:02}")
}

/// A stored `YYYY-MM-DD` (the ISO day of a file or a setting, a time may
/// follow it) as the interface shows it, `2026/10/22`; `None` when it does
/// not start with one.
pub fn numeric_date_of_iso(text: &str) -> Option<String> {
    let day = text.get(..10)?;
    let mut parts = day.split('-');
    let mut next = |len: usize| {
        let part = parts.next().filter(|p| p.len() == len && p.bytes().all(|b| b.is_ascii_digit()))?;
        part.parse::<i64>().ok()
    };
    let (year, month, day) = (next(4)?, next(2)?, next(2)?);
    ((1..=12).contains(&month) && (1..=31).contains(&day)).then(|| numeric_date(year, month, day))
}

/// The parts of a numeric date as the catalogue keys that compose one take
/// them (`$year`, `$month`, `$day`), padded like [`numeric_date`].
pub fn numeric_parts(year: i64, month: i64, day: i64) -> (String, String, String) {
    (format!("{year:04}"), format!("{month:02}"), format!("{day:02}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_dates_read_year_month_day_zero_padded() {
        assert_eq!(numeric_date(2026, 10, 15), "2026/10/15");
        assert_eq!(numeric_date(2026, 1, 5), "2026/01/05");
        assert_eq!(numeric_date(1970, 1, 1), "1970/01/01");
        assert_eq!(numeric_date(2024, 2, 29), "2024/02/29");
        assert_eq!(numeric_date(1999, 12, 31), "1999/12/31");
        assert_eq!(numeric_date(987, 6, 7), "0987/06/07");
        assert_eq!(numeric_date(10_000, 1, 1), "10000/01/01");
    }

    #[test]
    fn a_date_without_its_year_reads_month_then_day() {
        assert_eq!(numeric_month_day(10, 15), "10/15");
        assert_eq!(numeric_month_day(7, 5), "07/05");
        assert_eq!(numeric_month_day(12, 31), "12/31");
        assert_eq!(numeric_month_day(1, 1), "01/01");
    }

    #[test]
    fn a_stored_iso_day_reads_like_the_others() {
        assert_eq!(numeric_date_of_iso("2026-10-22").as_deref(), Some("2026/10/22"));
        assert_eq!(numeric_date_of_iso("2026-01-05T14:30:00Z").as_deref(), Some("2026/01/05"));
        assert_eq!(numeric_date_of_iso("2026-13-05"), None);
        assert_eq!(numeric_date_of_iso("2026-1-5"), None);
        assert_eq!(numeric_date_of_iso("10/22/2026"), None);
        assert_eq!(numeric_date_of_iso("2026-10"), None);
        assert_eq!(numeric_date_of_iso("é026-10-22"), None);
        assert_eq!(numeric_date_of_iso(""), None);
    }

    #[test]
    fn the_parts_are_padded_like_the_whole_date() {
        for (y, m, d) in [(2026, 10, 15), (2026, 1, 5), (1970, 1, 1), (987, 6, 7)] {
            let (year, month, day) = numeric_parts(y, m, d);
            assert_eq!(format!("{year}/{month}/{day}"), numeric_date(y, m, d));
            assert_eq!(format!("{month}/{day}"), numeric_month_day(m, d));
        }
    }
}
