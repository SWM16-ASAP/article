from ..state import BookState
from ..utils.logging_config import get_logger
from ..utils.workflow_helpers import is_custom_content, calculate_custom_progress, send_progress_update, send_discord_webhook
import os

logger = get_logger(__name__)

COST_THRESHOLD = 0.04

def log_final_summary(state: BookState) -> BookState:
    """Logs the total token usage and estimated cost per model."""

    logger.info("========== WORKFLOW COMPLETE ==========")

    for model_id, usage in state["usage_metrics"].items():

        logger.info(f"--- Model: {model_id} ---")
        logger.info(f"  Input Tokens:  {usage['input_tokens']:,}")
        logger.info(f"  Output Tokens: {usage['output_tokens']:,}")
        logger.info(f"  Estimated Cost: ${usage['cost']:.6f}")

    logger.info("------------------------------------")
    logger.info(f"Total Estimated Cost (All Models): ${state['total_cost']:.6f}")
    logger.info("======================================")

    # custom 콘텐츠일 때만 진행률 전송 및 완료 알림 전송
    try:
        if is_custom_content(state):
            # 비용 임계값 체크 및 디스코드 웹훅 전송
            if state['total_cost'] > COST_THRESHOLD:
                request_id = state.get("id", "Unknown")
                title = state.get("title", "제목 없음")
                
                message = f"""🚨 **커스텀 콘텐츠 비용 임계값 초과 알림** 🚨
                
                    **Request ID**: {request_id}
                    **제목**: {title}
                    **총 비용**: ${state['total_cost']:.6f}
                    **임계값**: ${COST_THRESHOLD:.2f}
                    **초과 금액**: ${state['total_cost'] - COST_THRESHOLD:.6f}

                    커스텀 콘텐츠 처리가 완료되었으나 비용이 설정된 임계값을 초과했습니다."""
                
                send_discord_webhook(message)
                logger.warning(f"비용 임계값 초과! Request ID: {request_id}, 총 비용: ${state['total_cost']:.6f}, 임계값: ${COST_THRESHOLD:.2f}")
            
            # 최종 진행률 전송
            progress = calculate_custom_progress(state, "final")
            send_progress_update(state, progress)
    except Exception:
        pass
    return state
