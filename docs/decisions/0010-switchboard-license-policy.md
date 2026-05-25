# ADR-0010: Switchboard(MSU transcripts) 라이선스 — 무제한, 단 다운로드 스크립트 + gitignore 유지

- **Status**: Accepted
- **Date**: 2026-05-26

## Context

보조 구어 코퍼스로 Switchboard를 [ADR-0009](0009-switchboard-turn-reconstruction.md)에 따라 도입한다. 확보처는 **OpenSLR #5 = "MSU Switchboard transcripts"**(ISIP/Mississippi State 재전사본)이다.

- OpenSLR #5 라이선스 표기(verbatim, openslr.org/5): **"Unrestricted"**, 설명문 **"These were released without any license restrictions."** 아카이브 내 `AAREADME.text`에는 별도 라이선스 절이 없고 파일 목록·생성 방법만 기술.
- 단, 원본 Switchboard-1 **오디오 및 LDC 배포본(LDC97S62)은 LDC 유료 라이선스**다. 우리가 쓰는 것은 OpenSLR이 무제한 배포하는 **MSU 텍스트 전사·단어 정렬·발음 사전뿐**이며 **오디오는 사용하지 않는다**.
- SBCSAE는 CC BY-ND 3.0이라 [ADR-0002](0002-sbcsae-license-policy.md)에서 git commit 금지 + 다운로드 스크립트 확보로 대응했다. Switchboard는 라이선스 성격이 다르므로 정책을 명시한다.

## Decision

- 사용 범위를 **OpenSLR #5의 MSU 텍스트 산출물로 한정**한다(LDC 오디오/배포본 미사용).
- 라이선스상 commit 제약은 없으나, **원본·산출물 모두 git commit하지 않고 다운로드 스크립트로 확보 + gitignore**한다 — SBCSAE와 동일하게 운영. 단 그 **이유는 라이선스 강제가 아니라 저장소 용량·운영 일관성**이다(원본 ~49MB, 수천 파일). 따라서 `datasets/switchboard/`, `data/*/Switchboard/`는 gitignore 대상.
- 인용/출처 표기: ISIP/MSU Switchboard transcripts, OpenSLR SLR5.

## Consequences

### 긍정적

- 라이선스 리스크 낮음("Unrestricted"). `datasets/` 운영 방식이 SBCSAE와 통일된다.

### 부정적 / 주의

- "Unrestricted"의 법적 출처 문서가 빈약하다(OpenSLR 한 줄 + ISIP 관행). **상업적 재배포 등 고위험 용도 전에는 별도 확인**이 필요하다. 현재 용도(연구·내부 학습 데이터 생성)에는 충분.

## Alternatives Considered

| 안 | 기각 이유 |
|---|---|
| LDC 정식 배포본(LDC97S62) 사용 | 유료·승인 필요. 텍스트만 필요한 현 단계엔 과함 |
| 무제한이므로 git에 commit | 용량·일관성 문제. `datasets/` gitignore 관행 유지가 단순하고 SBCSAE와 통일적 |

## References

- [ADR-0002](0002-sbcsae-license-policy.md) — SBCSAE 라이선스 대응
- [ADR-0009](0009-switchboard-turn-reconstruction.md) — Switchboard 턴 재구성
- openslr.org/5, `scripttuner/data_sources/` 다운로더 패턴